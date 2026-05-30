from pydub import AudioSegment
import os


def validate_audio(file_path):

    # Allowed formats
    allowed_formats = [".mp3", ".wav", ".ogg", ".m4a"]

    # Check format
    extension = os.path.splitext(file_path)[1].lower()

    if extension not in allowed_formats:
        print("❌ Unsupported format:", extension)
        return False

    # Load audio
    audio = AudioSegment.from_file(file_path)

    # Duration
    duration_seconds = len(audio) / 1000

    print(f"Duration  : {duration_seconds:.2f} sec")
    print(f"Channels  : {audio.channels}")
    print(f"Frame Rate: {audio.frame_rate} Hz")

    # Minimum duration check
    if duration_seconds < 5:
        print("❌ Audio too short — minimum 5 seconds")
        return False

    # Maximum duration check
    if duration_seconds > 350:
        print("❌ Audio too long — maximum 5 minutes")
        return False

    if duration_seconds > 250:
        print("⚠️ Warning: Long audio — processing may take time")

    print("✅ Audio valid")
    return True
