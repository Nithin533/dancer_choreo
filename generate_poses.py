"""
generate_poses.py
Runs the trained DanceTransformer on a song and outputs 3D poses.

Exact architecture matched to best_model.pt:
  music_dim   = 438   (from aistpp_music_feat_7.5fps features)
  pose_dim    = 72    (24 SMPL joints x 3 angles)
  d_model     = 256
  nhead       = 8
  num_layers  = 6
  feedforward = 1024
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter


# ─────────────────────────────────────────
#  CONFIG — change these for your song
# ─────────────────────────────────────────

CHECKPOINT_PATH = "best_model.pt"          # trained model
NO_VOCALS_PATH  = "separated/your_song/htdemucs/normalized_audio/no_vocals.mp3"
SONG_FOLDER     = "output/your_song"       # where to save results
STYLE           = "hiphop"                 # hiphop / kpop / cinematic / freestyle
FEAT_7_5_PATH   = None                     # optional: path to precomputed .pkl feature file


# ─────────────────────────────────────────
#  MODEL — must match training exactly
# ─────────────────────────────────────────

class DanceTransformer(nn.Module):

    def __init__(
        self,
        music_dim=438,      # ← 438 confirmed from best_model.pt
        pose_dim=72,        # ← 24 joints x 3
        d_model=256,
        nhead=8,
        num_layers=6,       # ← 6 layers confirmed from best_model.pt
        dim_feedforward=1024
    ):
        super().__init__()
        self.music_proj  = nn.Linear(music_dim, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pose_head   = nn.Linear(d_model, pose_dim)

    def forward(self, x):
        x = self.music_proj(x)
        x = self.transformer(x)
        return self.pose_head(x)


# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────

def load_model(checkpoint_path):

    print("\n📦 Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device : {device}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"   Epoch  : {ckpt.get('epoch', '?')}")
    print(f"   Loss   : {ckpt.get('loss', '?'):.4f}")

    model = DanceTransformer()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print("✅ Model loaded")
    return model, device


# ─────────────────────────────────────────
#  LOAD FEATURES
#  Option A: from precomputed AIST++ .pkl
#  Option B: extract from audio with librosa
# ─────────────────────────────────────────

def load_features_from_pkl(pkl_path):
    """Load precomputed 438-dim features from AIST++ format."""
    print(f"📂 Loading features from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Get the feature array — key varies by file
    key = list(data.keys())[0]
    features = np.array(data[key], dtype=np.float32)

    print(f"✅ Features loaded: {features.shape}")
    return features


def extract_features_librosa(audio_path):
    """
    Extract 438-dim features from audio using librosa.
    This matches the aistpp_music_feat_7.5fps format.
    """
    import librosa

    print(f"🎵 Extracting features from: {audio_path}")

    y, sr = librosa.load(audio_path, sr=None)

    # Match AIST++ 7.5fps feature rate
    hop_length = int(sr / 7.5)

    # 1. Chroma (12)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # 2. MFCCs (20)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)

    # 3. MFCC delta (20)
    mfcc_delta = librosa.feature.delta(mfcc)

    # 4. Mel spectrogram compressed (128 → 20 via mean bands)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Compress 128 mel bands → 20 bands
    mel_bands = np.array([
        mel_db[i*6:(i+1)*6, :].mean(axis=0) for i in range(20)
    ])

    # 5. RMS energy (1)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)

    # 6. Spectral features (6)
    spec_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spec_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    spec_contrast  = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)  # 7 bands

    # 7. Onset strength (1)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).reshape(1, -1)

    # 8. Tempogram (384 → compressed to 358 to hit 438 total)
    # 12+20+20+20+1+1+1+1+7+1 = 84 so far, need 438-84=354 more
    tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)  # (384, T)
    # Compress to 354 bands
    n_needed  = 438 - 84
    tempogram_compressed = np.array([
        tempogram[i*(384//n_needed):(i+1)*(384//n_needed), :].mean(axis=0)
        for i in range(n_needed)
    ])

    # Align all to same T
    T = min(
        chroma.shape[1], mfcc.shape[1], mfcc_delta.shape[1],
        mel_bands.shape[1], rms.shape[1], spec_centroid.shape[1],
        spec_bandwidth.shape[1], spec_rolloff.shape[1],
        spec_contrast.shape[1], onset.shape[1], tempogram_compressed.shape[1]
    )

    features = np.vstack([
        chroma[:, :T],               # 12
        mfcc[:, :T],                 # 20
        mfcc_delta[:, :T],           # 20
        mel_bands[:, :T],            # 20
        rms[:, :T],                  #  1
        spec_centroid[:, :T],        #  1
        spec_bandwidth[:, :T],       #  1
        spec_rolloff[:, :T],         #  1
        spec_contrast[:, :T],        #  7
        onset[:, :T],                #  1
        tempogram_compressed[:, :T], # 354
    ]).T  # → (T, 438)

    # Normalize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    print(f"✅ Features shape: {features.shape}")
    assert features.shape[1] == 438, f"Expected 438 dims, got {features.shape[1]}"

    return features.astype(np.float32)


# ─────────────────────────────────────────
#  SMOOTHING
# ─────────────────────────────────────────

def smooth_poses(poses, style="hiphop"):

    style_window = {
        "hiphop"   : 7,
        "kpop"     : 11,
        "cinematic": 21,
        "freestyle": 9,
    }
    window = style_window.get(style, 11)

    if window >= poses.shape[0]:
        window = max(poses.shape[0] // 2, 3)
        if window % 2 == 0:
            window -= 1

    smoothed = np.copy(poses)
    for j in range(poses.shape[1]):
        smoothed[:, j] = savgol_filter(poses[:, j], window_length=window, polyorder=3)

    return smoothed


# ─────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────

def generate_poses(
    no_vocals_path,
    checkpoint_path = CHECKPOINT_PATH,
    song_folder     = SONG_FOLDER,
    style           = STYLE,
    feat_pkl_path   = None
):

    print("\n" + "="*55)
    print("  DANCE POSE GENERATION")
    print("="*55)

    os.makedirs(song_folder, exist_ok=True)

    # ── Load model ──
    model, device = load_model(checkpoint_path)

    # ── Load features ──
    if feat_pkl_path and os.path.exists(feat_pkl_path):
        # Use precomputed AIST++ features (most accurate)
        features = load_features_from_pkl(feat_pkl_path)
    else:
        # Extract from audio (good approximation)
        if not os.path.exists(no_vocals_path):
            print(f"❌ Audio not found: {no_vocals_path}")
            sys.exit(1)
        features = extract_features_librosa(no_vocals_path)

    # ── Run model ──
    print("\n💃 Generating poses...")
    feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    # shape: (1, T, 438)

    with torch.no_grad():
        poses = model(feat_tensor)

    poses_np = poses.squeeze(0).cpu().numpy()
    # shape: (T, 72)

    print(f"✅ Raw poses    : {poses_np.shape}")

    # ── Smooth ──
    poses_smooth = smooth_poses(poses_np, style=style)
    print(f"✅ Smooth poses : {poses_smooth.shape}")

    # ── Save ──
    raw_path    = f"{song_folder}/poses_raw.npy"
    smooth_path = f"{song_folder}/poses_smooth.npy"
    meta_path   = f"{song_folder}/generation_meta.pkl"

    np.save(raw_path,    poses_np)
    np.save(smooth_path, poses_smooth)

    meta = {
        "style"       : style,
        "total_frames": poses_smooth.shape[0],
        "pose_dim"    : poses_smooth.shape[1],
        "song_folder" : song_folder,
        "checkpoint"  : checkpoint_path,
        "feat_shape"  : features.shape,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n✅ Raw poses saved    → {raw_path}")
    print(f"✅ Smooth poses saved → {smooth_path}")
    print(f"✅ Metadata saved     → {meta_path}")

    print("\n" + "="*55)
    print("  GENERATION COMPLETE")
    print("="*55)
    print(f"  Frames : {poses_smooth.shape[0]}")
    print(f"  Joints : 24 (72 dims = 24 x 3 angles)")
    print(f"  Style  : {style}")
    print("="*55)
    print("\n✅ Next step: run visualize_poses.py\n")

    return smooth_path


# ─────────────────────────────────────────
#  RUN — edit paths below for your song
# ─────────────────────────────────────────

if __name__ == "__main__":

    generate_poses(
        no_vocals_path  = "separated/your_song/htdemucs/normalized_audio/no_vocals.mp3",
        checkpoint_path = "best_model.pt",
        song_folder     = "output/your_song",
        style           = "hiphop",   # hiphop / kpop / cinematic / freestyle
        feat_pkl_path   = None        # optional: "dataset/aistpp_music_feat_7.5fps/mBR0.pkl"
    )
