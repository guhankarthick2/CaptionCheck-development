"""Extract speech with FFmpeg and transcribe it with faster-whisper."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("CAPTIONCHECK_CACHE", ROOT / "data" / "cache"))

_model = None


def ffmpeg_executable() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA", "")
    winget = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if winget.exists():
        matches = sorted(winget.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"))
        if matches:
            return str(matches[-1])
    raise RuntimeError("FFmpeg is required. Install it and open a new terminal so ffmpeg is on PATH.")


def extract_wav(source: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_executable(),
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )


def ffprobe_executable() -> str:
    ffmpeg = Path(ffmpeg_executable())
    probe = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    if probe.exists():
        return str(probe)
    found = shutil.which("ffprobe")
    if found:
        return found
    raise RuntimeError("ffprobe is required to check video length.")


def media_duration_seconds(source: Path) -> float:
    result = subprocess.run(
        [
            ffprobe_executable(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def convert_to_mp4(source: Path, dest: Path, max_seconds: float | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg_executable(), "-y", "-i", str(source)]
    if max_seconds is not None:
        command.extend(["-t", f"{max_seconds:.3f}"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )
    subprocess.run(command, check=True, capture_output=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel("small.en", device="cpu", compute_type="int8")
    return _model


def read_pcm_wav(wav_path: Path):
    """Load a 16 kHz mono wav as float32 samples for faster-whisper."""
    import numpy as np

    prepared = wav_path
    temporary = None
    with wave.open(str(wav_path)) as handle:
        needs_convert = (
            handle.getframerate() != 16000
            or handle.getnchannels() != 1
            or handle.getsampwidth() != 2
        )
    if needs_convert:
        temporary = wav_path.with_suffix(".16k.wav")
        extract_wav(wav_path, temporary)
        prepared = temporary
    with wave.open(str(prepared)) as handle:
        frames = handle.readframes(handle.getnframes())
    if temporary is not None:
        temporary.unlink(missing_ok=True)
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(audio)


def transcribe_wav(wav_path: Path) -> list[dict]:
    segments, _info = get_model().transcribe(
        read_pcm_wav(wav_path),
        language="en",
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    words = []
    for segment in segments:
        for word in segment.words or []:
            words.append(
                {
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "word": word.word.strip(),
                    "probability": round(float(word.probability), 4),
                }
            )
    return [word for word in words if word["word"]]


def load_or_transcribe(wav_path: Path, dest_json: Path) -> list[dict]:
    """Reuse words.json when this exact wav was transcribed before."""
    digest = file_sha256(wav_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{digest}.json"
    if cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
    else:
        payload = {"sha256": digest, "words": transcribe_wav(wav_path)}
        cached.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dest_json.parent.mkdir(parents=True, exist_ok=True)
    dest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload["words"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe a wav to words.json")
    parser.add_argument("wav", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    words = load_or_transcribe(args.wav, args.output)
    print(f"Wrote {len(words)} words to {args.output}")


if __name__ == "__main__":
    main()
