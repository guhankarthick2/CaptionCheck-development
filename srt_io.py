"""Read and write SubRip (SRT) caption files."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Cue:
    id: int
    start: float
    end: float
    text: str

    def to_dict(self, flagged: bool = False) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "flagged": flagged,
        }


_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def parse_timestamp(value: str) -> float:
    match = _TS.search(value.strip())
    if not match:
        raise ValueError(f"Bad timestamp: {value}")
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt(text: str) -> list[Cue]:
    cleaned = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", cleaned):
        lines = [line for line in block.split("\n") if line.strip() != ""]
        if not lines:
            continue
        time_index = 0
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            time_index = 1
        if "-->" not in lines[time_index]:
            continue
        start_raw, end_raw = lines[time_index].split("-->", 1)
        start = parse_timestamp(start_raw)
        end = parse_timestamp(end_raw)
        if end <= start:
            continue
        body = "\n".join(lines[time_index + 1 :]).strip()
        cue_id = len(cues) + 1
        if time_index == 1:
            try:
                cue_id = int(lines[0].strip())
            except ValueError:
                cue_id = len(cues) + 1
        cues.append(Cue(cue_id, start, end, body))
    return cues


def write_srt(cues: list[Cue]) -> str:
    ordered = sorted(cues, key=lambda cue: (cue.start, cue.id))
    blocks = []
    for index, cue in enumerate(ordered, start=1):
        blocks.append(
            f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text.strip()}\n"
        )
    return "\n".join(blocks).rstrip() + "\n"
