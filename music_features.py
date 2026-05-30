import librosa
import numpy as np
import os


def extract_features(no_vocals_path, song_folder="output"):

    print("🎼 Extracting music features...")

    y, sr = librosa.load(no_vocals_path)

    # BPM and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).flatten()[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Energy
    rms_energy = librosa.feature.rms(y=y)[0]

    # Chroma — harmonic content
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # MFCCs — timbral texture
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Song sections
    bounds = librosa.segment.agglomerative(chroma, 6)
    section_times = librosa.frames_to_time(bounds, sr=sr)

    print(f"✅ BPM          : {tempo:.1f}")
    print(f"✅ Energy frames: {len(rms_energy)}")
    print(f"✅ MFCC shape   : {mfcc.shape}")
    print(f"✅ Sections     : {[f'{t:.1f}s' for t in section_times]}")

    # Save features to song folder
    os.makedirs(song_folder, exist_ok=True)
    np.save(f"{song_folder}/beat_times.npy", beat_times)
    np.save(f"{song_folder}/rms_energy.npy", rms_energy)
    np.save(f"{song_folder}/mfcc.npy", mfcc)
    np.save(f"{song_folder}/section_times.npy", section_times)
    print(f"✅ Features saved → {song_folder}/")

    return {
        "tempo"        : tempo,
        "beat_times"   : beat_times,
        "rms_energy"   : rms_energy,
        "chroma"       : chroma,
        "mfcc"         : mfcc,
        "section_times": section_times
    }
