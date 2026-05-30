import librosa
import numpy as np
import os


def analyze_structure(no_vocals_path, song_folder="output"):

    print("🎵 Analyzing song structure...")

    y, sr = librosa.load(no_vocals_path)

    # Detect sections
    segments = librosa.effects.split(y, top_db=25)

    print(f"\n✅ Found {len(segments)} segments:\n")

    for i, segment in enumerate(segments):
        start = segment[0] / sr
        end = segment[1] / sr
        print(f"  Segment {i+1}: {start:.2f}s → {end:.2f}s")

    # Save segment data
    os.makedirs(song_folder, exist_ok=True)
    np.save(f"{song_folder}/segments.npy", segments)
    print(f"\n✅ Structure saved → {song_folder}/segments.npy")

    return segments
