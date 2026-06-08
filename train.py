"""
Dance Choreography AI — Training Script
Hardware: RTX 5070 8GB VRAM, Intel Core Ultra 9 285H, 32GB RAM

Features:
  - GPU/CPU/RAM guardrails (auto-reduces batch if VRAM fills up)
  - tqdm progress bars at every level
  - Smart checkpointing (every epoch if <10, every 10th if >=10)
  - Auto-resumes from last checkpoint if interrupted
"""

import os
import sys
import time
import pickle
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import gc

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─────────────────────────────────────────
#  GUARDRAIL SETTINGS — tuned for RTX 5070
# ─────────────────────────────────────────

VRAM_LIMIT_GB       = 7.0     # out of 8GB — leave 1GB headroom
RAM_LIMIT_GB        = 28.0    # out of 32GB — leave 4GB headroom
INITIAL_BATCH_SIZE  = 32      # will auto-reduce if VRAM fills up
MIN_BATCH_SIZE      = 4       # never go below this
NUM_EPOCHS          = 100
LEARNING_RATE       = 1e-4
NUM_WORKERS         = 4       # CPU workers for data loading
CHECKPOINT_DIR      = "checkpoints"
DATA_DIR            = "dataset/aistpp_music_feat_7.5fps"
MOTIONS_DIR         = "motions"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────
#  HARDWARE CHECK
# ─────────────────────────────────────────

def check_hardware():
    print("\n" + "=" * 55)
    print("  HARDWARE CHECK")
    print("=" * 55)

    # GPU
    if not torch.cuda.is_available():
        print("❌ No GPU found — training will be very slow on CPU")
        print("   Make sure CUDA 11.8 is installed")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        gpu_name  = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU     : {gpu_name}")
        print(f"✅ VRAM    : {vram_total:.1f} GB total | {VRAM_LIMIT_GB} GB limit set")

    # RAM
    ram_total = psutil.virtual_memory().total / 1e9
    ram_avail = psutil.virtual_memory().available / 1e9
    print(f"✅ RAM     : {ram_total:.1f} GB total | {ram_avail:.1f} GB available")
    print(f"✅ CPU     : {psutil.cpu_count(logical=False)} cores")

    if ram_avail < 8:
        print("⚠️  Warning: Low available RAM — close other programs")

    print("=" * 55 + "\n")
    return device


def get_vram_used_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e9
    return 0.0


def get_ram_used_gb():
    return psutil.virtual_memory().used / 1e9


def check_guardrails(batch_size):
    """Check VRAM and RAM — reduce batch size if needed."""
    vram_used = get_vram_used_gb()
    ram_used  = get_ram_used_gb()
    warned    = False

    if vram_used > VRAM_LIMIT_GB:
        new_batch = max(batch_size // 2, MIN_BATCH_SIZE)
        print(f"\n⚠️  VRAM guardrail hit: {vram_used:.2f}GB used")
        print(f"   Reducing batch size: {batch_size} → {new_batch}")
        torch.cuda.empty_cache()
        gc.collect()
        warned = True
        return new_batch, warned

    if ram_used > RAM_LIMIT_GB:
        print(f"\n⚠️  RAM guardrail hit: {ram_used:.2f}GB used")
        print(f"   Clearing cache...")
        gc.collect()
        warned = True

    return batch_size, warned


# ─────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────

class DanceDataset(Dataset):

    def __init__(self, music_dir, motions_dir):
        self.pairs = []

        print("📂 Loading dataset pairs...")

        # Music files are per-song: mBR4.json, mHO1.json, etc.
        music_files = sorted([
            f for f in os.listdir(music_dir)
            if f.endswith(".json")
        ])

        # Build lookup: music_id -> full path
        # e.g. "mBR4" -> "dataset/aistpp_music_feat_7.5fps/mBR4.json"
        music_lookup = {}
        for f in music_files:
            music_id = os.path.splitext(f)[0]          # "mBR4"
            music_lookup[music_id] = os.path.join(music_dir, f)

        # Load ignore list if present
        ignore = set()
        if os.path.exists("Bailando/ignore_list.txt"):
            with open("Bailando/ignore_list.txt") as f:
                ignore = set(line.strip() for line in f)
            print(f"   Ignoring {len(ignore)} bad sequences")

        # Motion files are per-sequence: gBR_sBM_cAll_d04_mBR4_ch02.pkl
        # Extract the music_id embedded in the filename (the mXXX part)
        motion_files = sorted([
            f for f in os.listdir(motions_dir)
            if f.endswith(".pkl")
        ])

        for motion_file in tqdm(motion_files, desc="  Pairing files"):
            seq_name = os.path.splitext(motion_file)[0]

            if seq_name in ignore:
                continue

            # Extract music_id from motion filename
            # Pattern: ..._{music_id}_ch##  e.g. mBR4, mHO1
            match = re.search(r'_(m[A-Z0-9]+)_ch', motion_file)
            if not match:
                continue

            music_id = match.group(1)   # "mBR4"

            if music_id not in music_lookup:
                continue

            self.pairs.append((
                music_lookup[music_id],
                os.path.join(motions_dir, motion_file)
            ))

        print(f"✅ Dataset: {len(self.pairs)} valid pairs found\n")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        music_path, motion_path = self.pairs[idx]

        with open(music_path, "r") as f:
            music = json.load(f)

        with open(motion_path, "rb") as f:
            motion = pickle.load(f)

        # Music features — shape (T, feature_dim)
        key = "music_array" if "music_array" in music else list(music.keys())[0]
        music_feat = torch.tensor(music[key], dtype=torch.float32)

        # Poses — shape (T, 24*3) — flatten joint angles
        poses = torch.tensor(
            motion["smpl_poses"], dtype=torch.float32
        )
        T = min(len(music_feat), len(poses))
        poses = poses[:T].reshape(T, -1)    # (T, 72)
        music_feat = music_feat[:T]         # (T, feat_dim)

        return music_feat, poses


def collate_fn(batch):
    """Handle variable length sequences."""
    music_list, pose_list = zip(*batch)

    # Pad to same length in batch
    max_len = max(m.shape[0] for m in music_list)

    music_padded = torch.zeros(len(music_list), max_len, music_list[0].shape[1])
    pose_padded  = torch.zeros(len(pose_list),  max_len, pose_list[0].shape[1])

    for i, (m, p) in enumerate(zip(music_list, pose_list)):
        music_padded[i, :m.shape[0]] = m
        pose_padded [i, :p.shape[0]] = p

    return music_padded, pose_padded


# ─────────────────────────────────────────
#  MODEL — Transformer based
# ─────────────────────────────────────────

# Inside train.py:
class DanceTransformer(nn.Module):
    def __init__(self, music_dim=438, pose_dim=72, d_model=256, nhead=8, num_layers=6):
        super().__init__()

        self.music_proj = nn.Linear(music_dim, d_model)

        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=1024, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.pose_head   = nn.Linear(d_model, pose_dim)

    def forward(self, music_feat):
        x = self.music_proj(music_feat)     # (B, T, d_model)
        x = self.transformer(x)             # (B, T, d_model)
        poses = self.pose_head(x)           # (B, T, 72)
        return poses


# ─────────────────────────────────────────
#  CHECKPOINTING
# ─────────────────────────────────────────

def get_checkpoint_interval(num_epochs):
    """Every epoch if <10, every 10th if >=10."""
    return 1 if num_epochs < 10 else 10


def save_checkpoint(epoch, model, optimizer, loss, batch_size, path):
    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "loss"       : loss,
        "batch_size" : batch_size,
    }, path)
    print(f"💾 Checkpoint saved → {path}")


def load_latest_checkpoint(model, optimizer):
    """Find and load the latest checkpoint."""
    checkpoints = sorted([
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("epoch_") and f.endswith(".pt")
    ])

    if not checkpoints:
        return 0, INITIAL_BATCH_SIZE, []

    latest  = checkpoints[-1]
    path    = os.path.join(CHECKPOINT_DIR, latest)
    ckpt    = torch.load(path, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])

    start_epoch = ckpt["epoch"] + 1
    batch_size  = ckpt.get("batch_size", INITIAL_BATCH_SIZE)
    loss        = ckpt["loss"]

    print(f"🔁 Resuming from epoch {ckpt['epoch']} | loss: {loss:.4f}")
    return start_epoch, batch_size, []


# ─────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────

def train():

    device = check_hardware()

    # ── Dataset ──
    print("[ STEP 1 ] Loading dataset...")
    dataset = DanceDataset(DATA_DIR, MOTIONS_DIR)

    if len(dataset) == 0:
        print("❌ No data found — check DATA_DIR and MOTIONS_DIR paths")
        sys.exit(1)

    # ── Model ──
    print("[ STEP 2 ] Building model...")
    model     = DanceTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {total_params:,}")

    # ── Resume from checkpoint ──
    print("\n[ STEP 3 ] Checking for checkpoints...")
    start_epoch, batch_size, loss_history = load_latest_checkpoint(
        model, optimizer
    )

    if start_epoch == 0:
        print("   No checkpoint found — starting fresh")
        loss_history = []

    ckpt_interval = get_checkpoint_interval(NUM_EPOCHS)
    print(f"✅ Checkpoint interval: every {ckpt_interval} epoch(s)")
    print(f"✅ Starting from epoch : {start_epoch}")
    print(f"✅ Batch size          : {batch_size}\n")

    # ── Training ──
    print("[ STEP 4 ] Training...\n")

    # Outer epoch bar
    epoch_bar = tqdm(
        range(start_epoch, NUM_EPOCHS),
        desc="Epochs",
        unit="epoch",
        initial=start_epoch,
        total=NUM_EPOCHS,
        colour="green"
    )

    for epoch in epoch_bar:

        # Rebuild dataloader (batch size may change)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda")
        )

        model.train()
        epoch_loss  = 0.0
        batch_count = 0

        # Inner batch bar
        batch_bar = tqdm(
            loader,
            desc=f"  Epoch {epoch+1:03d}/{NUM_EPOCHS}",
            unit="batch",
            leave=False,
            colour="blue"
        )

        for music, poses in batch_bar:

            music = music.to(device, non_blocking=True)
            poses = poses.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_poses = model(music)
            loss       = criterion(pred_poses, poses)
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss  += loss.item()
            batch_count += 1

            # Update batch bar
            batch_bar.set_postfix({
                "loss"    : f"{loss.item():.4f}",
                "VRAM_GB" : f"{get_vram_used_gb():.2f}",
                "RAM_GB"  : f"{get_ram_used_gb():.1f}",
            })

            # Check guardrails every 10 batches
            if batch_count % 10 == 0:
                batch_size, hit = check_guardrails(batch_size)
                if hit:
                    break   # restart epoch with new batch size

        avg_loss = epoch_loss / max(batch_count, 1)
        loss_history.append(avg_loss)

        # Update epoch bar
        epoch_bar.set_postfix({
            "avg_loss": f"{avg_loss:.4f}",
            "batch"   : batch_size,
            "VRAM_GB" : f"{get_vram_used_gb():.2f}",
        })

        # ── Checkpoint ──
        should_save = (
            (epoch + 1) % ckpt_interval == 0 or    # interval checkpoint
            (epoch + 1) == NUM_EPOCHS               # final epoch
        )

        if should_save:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR, f"epoch_{epoch+1:04d}.pt"
            )
            save_checkpoint(
                epoch, model, optimizer,
                avg_loss, batch_size, ckpt_path
            )

            # Also save as best if lowest loss
            if avg_loss == min(loss_history):
                save_checkpoint(
                    epoch, model, optimizer,
                    avg_loss, batch_size,
                    os.path.join(CHECKPOINT_DIR, "best_model.pt")
                )
                tqdm.write(f"🏆 New best model at epoch {epoch+1} | loss: {avg_loss:.4f}")

    # ── Done ──
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Best loss  : {min(loss_history):.4f}")
    print(f"  Checkpoint : {CHECKPOINT_DIR}/best_model.pt")
    print("=" * 55 + "\n")

    # Save loss history
    np.save(os.path.join(CHECKPOINT_DIR, "loss_history.npy"),
            np.array(loss_history))
    print("✅ Loss history saved → checkpoints/loss_history.npy")


if __name__ == "__main__":
    train()
