import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# ========= CONFIG =========
AUDIO_DIR = "/home/gpu/ragha/ragha/finai/data/demo/audio"
DELETE_ORIGINAL = True   # Set True if you want to remove .wav after conversion
QUALITY = "0"             # 0 = best, 2 = very high, 4 = good
# ==========================


def convert_file(wav_path: Path):
    mp3_path = wav_path.with_suffix(".mp3")

    if mp3_path.exists():
        print(f"Skipping (already exists): {mp3_path.name}")
        return

    command = [
        "ffmpeg",
        "-y",
        "-i", str(wav_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-qscale:a", QUALITY,
        str(mp3_path),
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        if DELETE_ORIGINAL:
            wav_path.unlink()

        print(f"Converted: {wav_path.name}")

    except subprocess.CalledProcessError:
        print(f"Failed: {wav_path.name}")


def main():
    wav_files = list(Path(AUDIO_DIR).rglob("*.wav"))

    if not wav_files:
        print("No WAV files found.")
        return

    print(f"Found {len(wav_files)} WAV files.")
    print(f"Using {multiprocessing.cpu_count()} threads.\n")

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(convert_file, wav_files)

    print("\nDone.")


if __name__ == "__main__":
    main()
