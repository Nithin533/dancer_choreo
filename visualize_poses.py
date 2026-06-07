"""
visualize_poses.py
Plays a live 3D stick figure animation synced to your song.

Requirements:
    pip install matplotlib numpy pygame scipy
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import threading
import time


# ─────────────────────────────────────────
#  CONFIG — edit these for your song
# ─────────────────────────────────────────

POSES_PATH   = "output/Raga/poses_smooth.npy"
META_PATH    = "output/Raga/generation_meta.pkl"
AUDIO_PATH   = "separated/Raga/htdemucs/normalized_audio/no_vocals.mp3"
FPS          = 7.5    # must match feature extraction rate


# ─────────────────────────────────────────
#  SMPL 24 JOINT SKELETON
#  defines which joints connect to which
# ─────────────────────────────────────────

JOINT_NAMES = [
    "pelvis",       # 0
    "left_hip",     # 1
    "right_hip",    # 2
    "spine1",       # 3
    "left_knee",    # 4
    "right_knee",   # 5
    "spine2",       # 6
    "left_ankle",   # 7
    "right_ankle",  # 8
    "spine3",       # 9
    "left_foot",    # 10
    "right_foot",   # 11
    "neck",         # 12
    "left_collar",  # 13
    "right_collar", # 14
    "head",         # 15
    "left_shoulder",# 16
    "right_shoulder",#17
    "left_elbow",   # 18
    "right_elbow",  # 19
    "left_wrist",   # 20
    "right_wrist",  # 21
    "left_hand",    # 22
    "right_hand",   # 23
]

# Bone connections — (parent, child)
BONES = [
    # Spine
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]

# Joint colors
JOINT_COLOR = {
    "spine" : "#00ffff",   # cyan
    "left"  : "#ff6b6b",   # red
    "right" : "#69ff47",   # green
    "head"  : "#ffffff",   # white
}

def get_joint_color(i):
    name = JOINT_NAMES[i]
    if "left"  in name: return "#ff6b6b"
    if "right" in name: return "#69ff47"
    if name in ["pelvis", "spine1", "spine2", "spine3", "neck", "head"]:
        return "#ffffff"
    return "#00ffff"


# ─────────────────────────────────────────
#  POSE → 3D POSITIONS
#  converts 72 joint angles → xyz positions
#  using a simple forward kinematics approximation
# ─────────────────────────────────────────

# Bone lengths (approximate, in meters)
BONE_LENGTHS = {
    (0,3): 0.10, (3,6): 0.10, (6,9): 0.10, (9,12): 0.10, (12,15): 0.15,
    (0,1): 0.10, (1,4): 0.40, (4,7): 0.40, (7,10): 0.10,
    (0,2): 0.10, (2,5): 0.40, (5,8): 0.40, (8,11): 0.10,
    (9,13): 0.12, (13,16): 0.15, (16,18): 0.28, (18,20): 0.25, (20,22): 0.08,
    (9,14): 0.12, (14,17): 0.15, (17,19): 0.28, (19,21): 0.25, (21,23): 0.08,
}

def angles_to_positions(pose_frame):
    """
    Convert 72 joint angles (24x3) → 24 xyz positions.
    pose_frame shape: (72,) → reshaped to (24, 3)
    """
    angles = pose_frame.reshape(24, 3)

    # Start pelvis at origin
    positions = np.zeros((24, 3))
    positions[0] = [0, 0, 0.9]   # pelvis height ~0.9m

    # Build skeleton using parent→child chain
    # Parent map for SMPL
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 9, 9,12,13,14,16,17,18,19,20,21]

    for i in range(1, 24):
        parent = parents[i]
        length = BONE_LENGTHS.get((parent, i), 0.15)

        # Use joint angles to determine direction
        angle = angles[i]

        # Convert angle to direction vector
        dx = np.sin(angle[1]) * np.cos(angle[0])
        dy = np.sin(angle[1]) * np.sin(angle[0])
        dz = np.cos(angle[1])

        direction = np.array([dx, dy, dz])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        positions[i] = positions[parent] + direction * length

    return positions


# ─────────────────────────────────────────
#  AUDIO PLAYBACK
# ─────────────────────────────────────────

def play_audio(audio_path):
    """Play audio in background thread."""
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        print("🎵 Audio playing...")
    except Exception as e:
        print(f"⚠️  Audio playback failed: {e}")
        print("   Install pygame: pip install pygame")
        print("   Continuing with silent animation...")


# ─────────────────────────────────────────
#  MAIN VISUALIZATION
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

    # ── Load poses ──
    if not os.path.exists(poses_path):
        print(f"❌ Poses not found: {poses_path}")
        print("   Run generate_poses.py first")
        sys.exit(1)

    poses = np.load(poses_path)
    print(f"✅ Poses loaded   : {poses.shape}")
    print(f"   Frames        : {poses.shape[0]}")
    print(f"   Joints x dims : {poses.shape[1]} (24 x 3)")

    # ── Load metadata ──
    style = "hiphop"
    tempo = 120.0
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        style = meta.get("style", "hiphop")
        tempo = meta.get("tempo", 120.0)
        print(f"   Style         : {style.upper()}")
        print(f"   BPM           : {tempo:.1f}")

    # ── Pre-compute all positions ──
    print("\n⚙️  Computing 3D positions...")
    all_positions = []
    for i in range(len(poses)):
        pos = angles_to_positions(poses[i])
        all_positions.append(pos)
    all_positions = np.array(all_positions)
    print(f"✅ Positions ready : {all_positions.shape}")

    # ── Setup figure ──
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")

    # ── Start audio ──
    if os.path.exists(audio_path):
        audio_thread = threading.Thread(
            target=play_audio,
            args=(audio_path,),
            daemon=True
        )
        audio_thread.start()
    else:
        print(f"⚠️  Audio not found: {audio_path}")
        print("   Running silent animation")

    # ── Animation ──
    interval_ms = int(1000 / fps)   # ms per frame

    def update(frame_idx):
        ax.cla()

        # Background
        ax.set_facecolor("black")

        positions = all_positions[frame_idx]  # (24, 3)

        # Draw bones
        for (j1, j2) in BONES:
            p1 = positions[j1]
            p2 = positions[j2]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color="#444444",
                linewidth=2.5,
                alpha=0.9
            )

        # Draw joints
        for i, pos in enumerate(positions):
            color = get_joint_color(i)
            size  = 80 if i in [0, 15] else 40   # bigger pelvis + head
            ax.scatter(
                pos[0], pos[1], pos[2],
                c=color, s=size, zorder=5
            )

        # Floor grid
        grid_range = np.linspace(-1, 1, 5)
        for g in grid_range:
            ax.plot([g, g], [-1, 1], [0, 0], color="#222222", linewidth=0.5)
            ax.plot([-1, 1], [g, g], [0, 0], color="#222222", linewidth=0.5)

        # Labels
        ax.set_title(
            f"Frame {frame_idx+1}/{len(poses)}  |  "
            f"Style: {style.upper()}  |  BPM: {tempo:.0f}",
            color="white", fontsize=11, pad=10
        )

        # Axis settings
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(0, 2.2)
        ax.set_xlabel("X", color="#555555")
        ax.set_ylabel("Y", color="#555555")
        ax.set_zlabel("Z", color="#555555")
        ax.tick_params(colors="#333333")

        # Rotate view slightly per frame for 3D effect
        ax.view_init(elev=15, azim=frame_idx * 0.3 % 360)

        return []

    print(f"\n▶  Starting animation at {fps}fps...")
    print("   Close the window to stop\n")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(poses),
        interval=interval_ms,
        blit=False,
        repeat=True
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    visualize(
        poses_path = POSES_PATH,
        audio_path = AUDIO_PATH,
        meta_path  = META_PATH,
        fps        = FPS
    )
