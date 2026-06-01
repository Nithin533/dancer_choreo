import torch
import torch.nn as nn
import numpy as np
import librosa
import os
import pickle


# ============================================================
#  MODEL DEFINITION
#  Must match exactly what you trained
# ============================================================

class DanceTransformer(nn.Module):

    def __init__(
        self,
        music_dim=35,       # input audio feature size
        pose_dim=72,        # output: 24 joints x 3 (SMPL)
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512
    ):
        super().__init__()

        # Project music features → model dimension
        self.music_proj = nn.Linear(music_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Project model output → pose
        self.pose_proj = nn.Linear(d_model, pose_dim)

    def forward(self, music_features):
        x = self.music_proj(music_features)
        x = self.transformer(x)
        poses = self.pose_proj(x)
        return poses


# ============================================================
#  LOAD CHECKPOINT
# ============================================================

def load_model(checkpoint_path):

    print("📦 Loading model from checkpoint...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    checkpoint = torch.load(
    os.path.join(checkpoint_path, "data.pkl"),
    map_location=device,
    weights_only=False
    )

    epoch = checkpoint.get("epoch", "unknown")
    print(f"   Trained epochs: {epoch}")

    # Build model and load weights
    model = DanceTransformer()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print("✅ Model loaded successfully")
    return model, device


# ============================================================
#  EXTRACT AUDIO FEATURES
# ============================================================

def extract_audio_features(no_vocals_path):

    print("🎵 Extracting audio features...")

    y, sr = librosa.load(no_vocals_path, sr=None)

    # BPM and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).flatten()[0])

    # MFCCs — 20 coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Chroma — 12 values
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Energy — 1 value
    rms = librosa.feature.rms(y=y)

    # Spectral centroid — 1 value
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Onset strength — 1 value
    onset = librosa.onset.onset_strength(y=y, sr=sr)

    # Align all to same length
    min_len = min(
        mfcc.shape[1],
        chroma.shape[1],
        rms.shape[1],
        centroid.shape[1],
        onset.shape[0]
    )

    # Stack into (T, 35) feature matrix
    features = np.vstack([
        mfcc[:, :min_len],          # 20
        chroma[:, :min_len],        # 12
        rms[:, :min_len],           #  1
        centroid[:, :min_len],      #  1
        onset[:min_len].reshape(1, -1)  # 1
    ]).T  # shape: (T, 35)

    # Normalize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    print(f"✅ BPM             : {tempo:.1f}")
    print(f"✅ Feature shape   : {features.shape}")

    return features, tempo


# ============================================================
#  GENERATE POSES
# ============================================================

def generate_poses(no_vocals_path, checkpoint_path, song_folder, style="hiphop"):

    print("\n" + "=" * 50)
    print("  POSE GENERATION")
    print("=" * 50 + "\n")

    # Style → smoothing window map
    style_smooth = {
        "hiphop"   : 7,
        "kpop"     : 11,
        "cinematic": 21,
        "freestyle": 9,
    }
    smooth_window = style_smooth.get(style, 11)
    print(f"🎨 Style: {style} (smooth window: {smooth_window})")

    # Load model
    model, device = load_model(checkpoint_path)

    # Extract features
    features, tempo = extract_audio_features(no_vocals_path)

    # Convert to tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    # shape: (1, T, 35)

    # Generate poses
    print("💃 Generating dance poses...")
    with torch.no_grad():
        poses = model(features_tensor)

    # Convert to numpy — shape: (T, 72)
    poses_np = poses.squeeze(0).cpu().numpy()
    print(f"✅ Raw poses shape : {poses_np.shape}")

    # Smooth poses
    poses_smooth = smooth_poses(poses_np, window=smooth_window)
    print(f"✅ Smoothed poses  : {poses_smooth.shape}")

    # Save
    os.makedirs(song_folder, exist_ok=True)
    raw_path    = f"{song_folder}/poses_raw.npy"
    smooth_path = f"{song_folder}/poses_smooth.npy"

    np.save(raw_path,    poses_np)
    np.save(smooth_path, poses_smooth)

    print(f"✅ Raw poses saved    → {raw_path}")
    print(f"✅ Smooth poses saved → {smooth_path}")

    # Save metadata
    meta = {
        "style"       : style,
        "tempo"       : tempo,
        "total_frames": poses_smooth.shape[0],
        "pose_dim"    : poses_smooth.shape[1],
        "song_folder" : song_folder,
    }
    meta_path = f"{song_folder}/generation_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Metadata saved     → {meta_path}")

    print("\n" + "=" * 50)
    print("  GENERATION COMPLETE")
    print("=" * 50)
    print(f"  Total frames : {poses_smooth.shape[0]}")
    print(f"  Pose dim     : {poses_smooth.shape[1]} (24 joints × 3)")
    print(f"  Style        : {style}")
    print(f"  BPM          : {tempo:.1f}")
    print("=" * 50 + "\n")
    print("✅ Ready for 3D rendering (Day 6)")

    return smooth_path


# ============================================================
#  MOTION SMOOTHING
# ============================================================

def smooth_poses(poses, window=11):

    from scipy.signal import savgol_filter

    # window must be odd and less than data length
    if window >= poses.shape[0]:
        window = poses.shape[0] // 2
        if window % 2 == 0:
            window -= 1

    smoothed = np.copy(poses)
    for joint in range(poses.shape[1]):
        smoothed[:, joint] = savgol_filter(
            poses[:, joint],
            window_length=window,
            polyorder=3
        )

    return smoothed


# ============================================================
#  RUN
# ============================================================

generate_poses(
    no_vocals_path  = "separated/test-1/htdemucs/normalized_audio/no_vocals.mp3",
    checkpoint_path = "best_model",
    song_folder     = "output/test-1",
    style           = "hiphop"   # change to: hiphop / kpop / cinematic / freestyle
)
