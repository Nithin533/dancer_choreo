from pydub import AudioSegment
from pydub.effects import normalize
import os
import subprocess

from audio_validation import validate_audio
from beat_detection import detect_beats
from music_features import extract_features
from song_structure import analyze_structure


def process_audio(file_path):

    print("\n" + "=" * 50)
    print("  DANCE CHOREOGRAPHY PIPELINE")
    print("=" * 50 + "\n")

    # ---------- Step 1: Validate ----------
    print("[ STEP 1 ] Validating audio...")
    if not validate_audio(file_path):
        return None

    # ---------- Step 2: Normalize ----------
    print("\n[ STEP 2 ] Normalizing audio...")

    song_name   = os.path.splitext(os.path.basename(file_path))[0]
    song_folder = f"output/{song_name}"
    os.makedirs(song_folder, exist_ok=True)

    audio            = AudioSegment.from_file(file_path)
    normalized_audio = normalize(audio)
    normalized_path  = f"{song_folder}/normalized_audio.wav"
    normalized_audio.export(normalized_path, format="wav")

    print("✅ Audio normalized")
    print(f"📂 Output folder: {song_folder}")

    # ---------- Step 3: Remove vocals ----------
    print("\n[ STEP 3 ] Removing vocals with Demucs...")

    separated_folder = f"separated/{song_name}"

    subprocess.run([
        "demucs",
        "--two-stems=vocals",
        "--mp3",
        "--out", separated_folder,
        normalized_path
    ])

    no_vocals_path = f"{separated_folder}/htdemucs/normalized_audio/no_vocals.mp3"

    if not os.path.exists(no_vocals_path):
        print("❌ Demucs output not found — check separated/ folder")
        return None

    print("✅ Vocals removed")
    print(f"📂 No-vocals: {no_vocals_path}")

    # ---------- Step 4: Detect beats ----------
    print("\n[ STEP 4 ] Detecting beats...")
    detect_beats(no_vocals_path, song_folder)

    # ---------- Step 5: Extract features ----------
    print("\n[ STEP 5 ] Extracting music features...")
    extract_features(no_vocals_path, song_folder)

    # ---------- Step 6: Song structure ----------
    print("\n[ STEP 6 ] Analyzing song structure...")
    analyze_structure(no_vocals_path, song_folder)

    # ---------- Done ----------
    print("\n" + "=" * 50)
    print("  PIPELINE COMPLETE")
    print("=" * 50)
    print(f"  Song folder : {song_folder}/")
    print(f"  No-vocals   : {no_vocals_path}")
    print("=" * 50 + "\n")
    print("✅ Ready for dance generation")

    # Return plain string — just the no_vocals path
    return no_vocals_path


if __name__ == "__main__":
    import sys
    song = sys.argv[1] if len(sys.argv) > 1 else "audio/test-1.mp3"
    process_audio(song)
