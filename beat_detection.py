import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def detect_beats(no_vocals_path, song_folder="output"):

    print("🥁 Detecting beats...")

    y, sr = librosa.load(no_vocals_path)

    # Main beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).flatten()[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Sub-beats — midpoint between each beat pair
    sub_beat_times = []
    for i in range(len(beat_times) - 1):
        mid = (beat_times[i] + beat_times[i + 1]) / 2
        sub_beat_times.append(mid)

    # Onsets — accent moments
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    print(f"✅ BPM        : {tempo:.1f}")
    print(f"✅ Beats      : {len(beat_times)}")
    print(f"✅ Sub-beats  : {len(sub_beat_times)}")
    print(f"✅ Onsets     : {len(onset_times)}")

    # Visualize
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)

    for beat in beat_times:
        plt.axvline(beat, color='red', alpha=0.5, linewidth=0.8)

    plt.title("Detected Beats — Red: beats")
    plt.tight_layout()

    os.makedirs(song_folder, exist_ok=True)
    plt.savefig(f"{song_folder}/beat_map.png")
    print(f"✅ Beat map saved → {song_folder}/beat_map.png")
    plt.show()

    return beat_times, sub_beat_times, onset_times, tempo
