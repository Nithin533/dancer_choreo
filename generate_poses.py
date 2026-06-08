"""
generate_poses.py
Runs the trained DanceTransformer on a song and outputs 3D poses.
Style is automatically detected from the audio.

Exact architecture matched to best_model.pt:
  music_dim   = 438
  pose_dim    = 72
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
import librosa
from scipy.signal import savgol_filter


# ─────────────────────────────────────────
#  MODEL — must match training exactly
# ─────────────────────────────────────────

class DanceTransformer(nn.Module):

    def __init__(
        self,
        music_dim=438,
        pose_dim=72,
        d_model=256,
        nhead=8,
        num_layers=6,
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
#  STYLE DETECTION
# ─────────────────────────────────────────

def detect_style(audio_path):

    print("\n🎧 Detecting style from audio...")

    y, sr = librosa.load(audio_path)

    tempo, _     = librosa.beat.beat_track(y=y, sr=sr)
    tempo        = float(np.asarray(tempo).flatten()[0])
    avg_energy   = float(np.mean(librosa.feature.rms(y=y)[0]))
    avg_flux     = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
    avg_rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))

    print(f"   BPM    : {tempo:.1f}")
    print(f"   Energy : {avg_energy:.4f}")
    print(f"   Flux   : {avg_flux:.4f}")
    print(f"   Rolloff: {avg_rolloff:.1f} Hz")

    if tempo > 120 and avg_energy > 0.05 and avg_flux > 2.0:
        style  = "hiphop"
        reason = f"fast tempo ({tempo:.0f} BPM) + high energy"
    elif tempo > 105 and avg_energy > 0.03 and avg_rolloff > 3000:
        style  = "kpop"
        reason = f"upbeat tempo ({tempo:.0f} BPM) + bright sound"
    elif tempo < 90 and avg_energy < 0.03:
        style  = "cinematic"
        reason = f"slow tempo ({tempo:.0f} BPM) + low energy"
    else:
        style  = "freestyle"
        reason = f"mixed tempo ({tempo:.0f} BPM)"

    print(f"\n✅ Detected style : {style.upper()}")
    print(f"   Reason        : {reason}")

    return style, tempo


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

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"   Epoch  : {ckpt.get('epoch', '?')}")
    print(f"   Loss   : {ckpt.get('loss', '?'):.4f}")

    model = DanceTransformer()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print("✅ Model loaded")
    return model, device


# ─────────────────────────────────────────
#  FEATURE EXTRACTION — 438 dims
# ─────────────────────────────────────────

def extract_features(audio_path):

    print(f"\n🎵 Extracting 438-dim features...")

    y, sr      = librosa.load(audio_path, sr=None)
    hop_length = int(sr / 7.5)

    chroma         = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc           = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    mfcc_delta     = librosa.feature.delta(mfcc)
    mel            = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
    mel_db         = librosa.power_to_db(mel, ref=np.max)
    mel_bands      = np.array([mel_db[i*6:(i+1)*6, :].mean(axis=0) for i in range(20)])
    rms            = librosa.feature.rms(y=y, hop_length=hop_length)
    spec_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spec_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    spec_contrast  = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    onset          = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).reshape(1, -1)
    tempogram      = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)
    n_needed       = 354
    tempogram_compressed = np.array([
        tempogram[i*(384//n_needed):(i+1)*(384//n_needed), :].mean(axis=0)
        for i in range(n_needed)
    ])

    T = min(
        chroma.shape[1], mfcc.shape[1], mfcc_delta.shape[1],
        mel_bands.shape[1], rms.shape[1], spec_centroid.shape[1],
        spec_bandwidth.shape[1], spec_rolloff.shape[1],
        spec_contrast.shape[1], onset.shape[1],
        tempogram_compressed.shape[1]
    )

    features = np.vstack([
        chroma[:, :T],
        mfcc[:, :T],
        mfcc_delta[:, :T],
        mel_bands[:, :T],
        rms[:, :T],
        spec_centroid[:, :T],
        spec_bandwidth[:, :T],
        spec_rolloff[:, :T],
        spec_contrast[:, :T],
        onset[:, :T],
        tempogram_compressed[:, :T],
    ]).T  # (T, 438)

    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    print(f"✅ Features shape : {features.shape}")
    assert features.shape[1] == 438, f"Expected 438 dims, got {features.shape[1]}"

    return features.astype(np.float32)


# ─────────────────────────────────────────
#  SMOOTHING
# ─────────────────────────────────────────

def smooth_poses(poses, style):

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

    print(f"✅ Smoothing      : window={window} (style={style})")
    return smoothed


# ─────────────────────────────────────────
#  MAIN — called by api.py automatically
# ─────────────────────────────────────────

def generate_poses(
    no_vocals_path,
    checkpoint_path = "best_model.pt",
    song_folder     = None,
):
    print("\n" + "="*55)
    print("  DANCE POSE GENERATION")
    print("="*55)

    # Auto-derive song_folder from no_vocals_path if not given
    # e.g. "separated/Raga/htdemucs/..." → song_name = "Raga"
    if song_folder is None:
        parts     = no_vocals_path.replace("\\", "/").split("/")
        song_name = parts[1] if len(parts) > 1 else "output"
        song_folder = f"output/{song_name}"

    os.makedirs(song_folder, exist_ok=True)

    if not os.path.exists(no_vocals_path):
        print(f"❌ Audio not found: {no_vocals_path}")
        return None

    # Auto detect style
    style, tempo = detect_style(no_vocals_path)

    # Load model
    model, device = load_model(checkpoint_path)

    # Extract features
    features    = extract_features(no_vocals_path)
    feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

    # Generate
    print("\n💃 Generating poses...")
    with torch.no_grad():
        poses = model(feat_tensor)

    poses_np = poses.squeeze(0).cpu().numpy()
    print(f"✅ Raw poses      : {poses_np.shape}")

    # Smooth
    poses_smooth = smooth_poses(poses_np, style=style)

    # Save
    raw_path    = f"{song_folder}/poses_raw.npy"
    smooth_path = f"{song_folder}/poses_smooth.npy"
    meta_path   = f"{song_folder}/generation_meta.pkl"

    np.save(raw_path,    poses_np)
    np.save(smooth_path, poses_smooth)

    meta = {
        "style"       : style,
        "tempo"       : tempo,
        "total_frames": poses_smooth.shape[0],
        "pose_dim"    : poses_smooth.shape[1],
        "song_folder" : song_folder,
        "checkpoint"  : checkpoint_path,
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
    print(f"  Style  : {style.upper()} (auto detected)")
    print(f"  BPM    : {tempo:.1f}")
    print(f"  Joints : 24 (72 dims = 24 x 3 angles)")
    print("="*55 + "\n")

    return smooth_path


# ─────────────────────────────────────────
#  RUN STANDALONE
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "separated/Raga/htdemucs/normalized_audio/no_vocals.mp3"
    generate_poses(no_vocals_path=path)
