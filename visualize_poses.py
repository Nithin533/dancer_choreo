"""
visualize_poses.py — Live 3D stick figure synced to music
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import threading


# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────

POSES_PATH = "output/Raga/poses_smooth.npy"
META_PATH  = "output/Raga/generation_meta.pkl"
AUDIO_PATH = "separated/Raga/htdemucs/normalized_audio/no_vocals.mp3"
FPS        = 7.5


# ─────────────────────────────────────────
#  SMPL SKELETON — T-pose joint positions
#  These are the resting positions of each
#  joint before any movement is applied
# ─────────────────────────────────────────

# Base T-pose positions (x, y, z) in meters
T_POSE = np.array([
    [ 0.00,  0.00,  0.00],   #  0 pelvis
    [-0.10,  0.00, -0.05],   #  1 left_hip
    [ 0.10,  0.00, -0.05],   #  2 right_hip
    [ 0.00,  0.00,  0.10],   #  3 spine1
    [-0.10,  0.00, -0.45],   #  4 left_knee
    [ 0.10,  0.00, -0.45],   #  5 right_knee
    [ 0.00,  0.00,  0.20],   #  6 spine2
    [-0.10,  0.00, -0.85],   #  7 left_ankle
    [ 0.10,  0.00, -0.85],   #  8 right_ankle
    [ 0.00,  0.00,  0.30],   #  9 spine3
    [-0.10,  0.00, -0.95],   # 10 left_foot
    [ 0.10,  0.00, -0.95],   # 11 right_foot
    [ 0.00,  0.00,  0.45],   # 12 neck
    [-0.10,  0.00,  0.40],   # 13 left_collar
    [ 0.10,  0.00,  0.40],   # 14 right_collar
    [ 0.00,  0.00,  0.60],   # 15 head
    [-0.35,  0.00,  0.38],   # 16 left_shoulder
    [ 0.35,  0.00,  0.38],   # 17 right_shoulder
    [-0.65,  0.00,  0.38],   # 18 left_elbow
    [ 0.65,  0.00,  0.38],   # 19 right_elbow
    [-0.90,  0.00,  0.38],   # 20 left_wrist
    [ 0.90,  0.00,  0.38],   # 21 right_wrist
    [-1.00,  0.00,  0.38],   # 22 left_hand
    [ 1.00,  0.00,  0.38],   # 23 right_hand
], dtype=np.float32)

# Lift the whole skeleton so feet are at z=0
T_POSE[:, 2] -= T_POSE[:, 2].min()
T_POSE[:, 2] += 0.05   # small floor gap

# Bone connections
BONES = [
    (0,1),(0,2),(0,3),         # pelvis
    (1,4),(4,7),(7,10),        # left leg
    (2,5),(5,8),(8,11),        # right leg
    (3,6),(6,9),               # spine
    (9,12),(12,15),            # neck + head
    (9,13),(13,16),            # left shoulder
    (9,14),(14,17),            # right shoulder
    (16,18),(18,20),(20,22),   # left arm
    (17,19),(19,21),(21,23),   # right arm
]

JOINT_NAMES = [
    "pelvis","left_hip","right_hip","spine1",
    "left_knee","right_knee","spine2",
    "left_ankle","right_ankle","spine3",
    "left_foot","right_foot","neck",
    "left_collar","right_collar","head",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hand","right_hand"
]

def get_color(i):
    name = JOINT_NAMES[i]
    if "left"  in name: return "#ff6b6b"
    if "right" in name: return "#69ff47"
    if name == "head":  return "#ffff00"
    return "#ffffff"

def get_bone_color(j1, j2):
    n1 = JOINT_NAMES[j1]
    n2 = JOINT_NAMES[j2]
    if "left"  in n1 or "left"  in n2: return "#cc4444"
    if "right" in n1 or "right" in n2: return "#44aa44"
    return "#8888ff"


# ─────────────────────────────────────────
#  POSE CONVERSION
#  poses are (T, 72) → reshape to (T, 24, 3)
#  each value is a rotation angle
#  we add it as an offset to T-pose positions
# ─────────────────────────────────────────

def pose_to_positions(pose_frame, scale=0.3):
    """
    pose_frame: (72,) array of joint angles
    Returns: (24, 3) array of 3D positions
    """
    angles = pose_frame.reshape(24, 3)

    # Start from T-pose
    positions = T_POSE.copy()

    # Apply rotations as positional offsets
    # Scale controls how much the angles move joints
    positions[:, 0] += np.sin(angles[:, 1]) * scale * 0.5
    positions[:, 1] += np.sin(angles[:, 2]) * scale * 0.3
    positions[:, 2] += np.sin(angles[:, 0]) * scale * 0.4

    # Keep feet near ground
    min_z = positions[[7,8,10,11], 2].min()
    if min_z < 0:
        positions[:, 2] -= min_z

    return positions


# ─────────────────────────────────────────
#  AUDIO
# ─────────────────────────────────────────

def play_audio(audio_path):
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        print("🎵 Audio playing...")
    except Exception as e:
        print(f"⚠️  Audio failed: {e}")


# ─────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────

def visualize(
    poses_path = POSES_PATH,
    audio_path = AUDIO_PATH,
    meta_path  = META_PATH,
    fps        = FPS
):
    print("\n" + "="*55)
    print("  3D DANCE VISUALIZATION")
    print("="*55 + "\n")

    # Load poses
    if not os.path.exists(poses_path):
        print(f"❌ Poses not found: {poses_path}")
        sys.exit(1)

    poses = np.load(poses_path)
    print(f"✅ Poses loaded  : {poses.shape}")

    # Load metadata
    style = "hiphop"
    tempo = 120.0
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        style = meta.get("style", "hiphop")
        tempo = meta.get("tempo", 120.0)

    print(f"   Style         : {style.upper()}")
    print(f"   BPM           : {tempo:.1f}")

    # Pre-compute all positions
    print("⚙️  Computing positions...")
    all_pos = np.array([pose_to_positions(poses[i]) for i in range(len(poses))])
    print(f"✅ Ready          : {all_pos.shape}")

    # Setup figure
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(9, 10), facecolor="black")
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Start audio
    if os.path.exists(audio_path):
        threading.Thread(target=play_audio, args=(audio_path,), daemon=True).start()
    else:
        print(f"⚠️  Audio not found: {audio_path}")

    interval_ms = int(1000 / fps)

    def update(frame_idx):
        ax.cla()
        ax.set_facecolor("black")

        pos = all_pos[frame_idx]   # (24, 3)

        # Draw bones
        for (j1, j2) in BONES:
            color = get_bone_color(j1, j2)
            ax.plot(
                [pos[j1,0], pos[j2,0]],
                [pos[j1,1], pos[j2,1]],
                [pos[j1,2], pos[j2,2]],
                color=color, linewidth=3, alpha=0.9
            )

        # Draw joints
        for i in range(24):
            color = get_color(i)
            size  = 120 if i == 15 else 60   # bigger head
            ax.scatter(
                pos[i,0], pos[i,1], pos[i,2],
                c=color, s=size, zorder=5, depthshade=False
            )

        # Floor
        for g in np.linspace(-1.2, 1.2, 7):
            ax.plot([g,g],[-1.2,1.2],[0,0], color="#1a1a1a", lw=0.5)
            ax.plot([-1.2,1.2],[g,g],[0,0], color="#1a1a1a", lw=0.5)

        # Title
        ax.set_title(
            f"Frame {frame_idx+1}/{len(poses)}   {style.upper()}   {tempo:.0f} BPM",
            color="white", fontsize=11
        )

        # Axis
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(0,   2.2)
        ax.set_xlabel("", color="#222")
        ax.set_ylabel("", color="#222")
        ax.set_zlabel("", color="#222")
        ax.tick_params(colors="#222222")
        ax.grid(False)

        # Slowly rotate
        ax.view_init(elev=12, azim=(frame_idx * 0.5) % 360)

        return []

    print(f"\n▶  Playing at {fps}fps — close window to stop\n")

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(poses),
        interval=interval_ms,
        blit=False,
        repeat=True
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()
