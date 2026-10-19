"""Build the planted demo clip with Windows speech synthesis."""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from srt_io import Cue, write_srt
from transcribe import ffmpeg_executable

ROOT = Path(__file__).resolve().parent
SENTENCES = [
    "Do not mix these chemicals.",
    "Use zero point zero five milligrams.",
    "Rinse with water right away.",
]
CAPTIONS = [
    "Mix these chemicals.",
    "Use 0.5 milligrams.",
]
GAP_SECONDS = 1.2


def synthesize(path: Path, text: str) -> None:
    safe_path = str(path.resolve()).replace("'", "''")
    safe_text = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$synth.Rate = -3; "
        f"$synth.SetOutputToWaveFile('{safe_path}'); "
        f"$synth.Speak('{safe_text}'); "
        "$synth.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)


def duration(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def concat(paths: list[Path], dest: Path, gap: float) -> list[float]:
    with wave.open(str(paths[0])) as first:
        params = first.getparams()
        rate = first.getframerate()
        width = first.getsampwidth()
        channels = first.getnchannels()
    silence = b"\x00" * int(gap * rate) * width * channels
    starts = []
    cursor = 0.0
    with wave.open(str(dest), "w") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        for index, path in enumerate(paths):
            if index:
                output.writeframes(silence)
                cursor += gap
            starts.append(cursor)
            with wave.open(str(path)) as handle:
                if handle.getframerate() != rate or handle.getsampwidth() != width:
                    raise RuntimeError(f"Unexpected wav format: {path}")
                frames = handle.readframes(handle.getnframes())
                output.writeframes(frames)
                cursor += handle.getnframes() / rate
    return starts


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, sentence in enumerate(SENTENCES, start=1):
        path = ROOT / f"part{index}.wav"
        synthesize(path, sentence)
        parts.append(path)
    lengths = [duration(path) for path in parts]
    starts = concat(parts, ROOT / "demo.wav", GAP_SECONDS)
    cues = []
    for index, (caption, start, length) in enumerate(zip(CAPTIONS, starts, lengths), start=1):
        cues.append(Cue(index, max(0.0, start), start + length + 0.25, caption))
    (ROOT / "demo.srt").write_text(write_srt(cues), encoding="utf-8")
    with wave.open(str(ROOT / "demo.wav")) as handle:
        total = handle.getnframes() / handle.getframerate()
    subprocess.run(
        [
            ffmpeg_executable(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x24312f:s=1280x720:r=30:d={total + 0.2:.3f}",
            "-i",
            str(ROOT / "demo.wav"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(ROOT / "demo.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    print("starts", [round(value, 3) for value in starts])
    print("lengths", [round(value, 3) for value in lengths])
    print("wrote", ROOT / "demo.wav")
    print("wrote", ROOT / "demo.srt")


if __name__ == "__main__":
    main()
