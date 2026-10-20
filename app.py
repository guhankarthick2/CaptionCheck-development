"""Local review server for CaptionCheck."""

from __future__ import annotations

import json
import os
import shutil
import traceback
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from flags import audio_text, build_flags, normalize_caption_text, words_in_cue
from srt_io import Cue, parse_srt, write_srt
from transcribe import convert_to_mp4, extract_wav, load_or_transcribe, media_duration_seconds

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CAPTIONCHECK_DATA", ROOT / "data" / "sessions"))
SAMPLES = ROOT / "samples"
MAX_SECONDS = float(os.environ.get("CAPTIONCHECK_MAX_SECONDS", "90"))
ALLOWED_VIDEO = {".mp4", ".webm", ".mov", ".mkv"}
LIMIT = (
    "If the second transcript hears the same wrong word as the caption, that line will not appear. "
    "Every row is a possible issue. Clips longer than "
    f"{int(MAX_SECONDS)} seconds are trimmed."
)

app = FastAPI(title="CaptionCheck")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class TextBody(BaseModel):
    text: str


class FlagBody(BaseModel):
    flagId: str
    text: str | None = None


def session_dir(session_id: str) -> Path:
    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(status_code=404, detail="Unknown session")
    path = DATA / session_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown session")
    return path


def load_state(path: Path) -> dict:
    return json.loads((path / "state.json").read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    (path / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def cues_from_state(state: dict) -> list[Cue]:
    return [Cue(cue["id"], cue["start"], cue["end"], cue["text"]) for cue in state["cues"]]


def public_state(path: Path, state: dict) -> dict:
    status = state.get("status", "ready")
    base = {
        "id": state["id"],
        "status": status,
        "progress": state.get("progress", ""),
        "limit": LIMIT,
        "glossary": state.get("glossary") or [],
        "error": state.get("error"),
        "maxSeconds": MAX_SECONDS,
    }
    if status != "ready":
        return base
    cues = cues_from_state(state)
    flags = build_flags(
        cues,
        state["words"],
        state.get("glossary") or [],
        set(state.get("dismissed") or []),
    )
    flagged_ids = {flag["cueId"] for flag in flags if flag["cueId"] is not None}
    cue_rows = []
    for cue in cues:
        row = cue.to_dict(flagged=cue.id in flagged_ids)
        row["audioText"] = audio_text(words_in_cue(state["words"], cue))
        cue_rows.append(row)
    return {
        **base,
        "cues": cue_rows,
        "flags": flags,
        "videoUrl": f"/api/sessions/{state['id']}/video",
        "emptySpeech": not state["words"],
        "trimmed": bool(state.get("trimmed")),
        "duration": state.get("duration"),
    }


def decode_srt(srt_bytes: bytes) -> list[Cue]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            text = srt_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise HTTPException(status_code=400, detail="The SRT file must be readable text.")
    try:
        cues = parse_srt(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not cues:
        raise HTTPException(status_code=400, detail="The SRT file has no caption cues.")
    for cue in cues:
        cue.text = normalize_caption_text(cue.text)
    return cues


def set_progress(path: Path, state: dict, message: str) -> None:
    state["progress"] = message
    save_state(path, state)


def process_session(session_id: str) -> None:
    path = DATA / session_id
    state = load_state(path)
    try:
        set_progress(path, state, "Checking video length…")
        source = next(path.glob("upload.*"))
        duration = media_duration_seconds(source)
        state["duration"] = round(duration, 2)
        trimmed = duration > MAX_SECONDS
        state["trimmed"] = trimmed

        set_progress(path, state, "Preparing video…")
        video_path = path / "video.mp4"
        if source.suffix.lower() == ".mp4" and not trimmed:
            shutil.copyfile(source, video_path)
        else:
            convert_to_mp4(source, video_path, MAX_SECONDS if trimmed else None)

        set_progress(path, state, "Extracting audio…")
        wav_path = path / "audio.wav"
        extract_wav(video_path, wav_path)

        set_progress(path, state, "Transcribing speech. The first run downloads the model…")
        words = load_or_transcribe(wav_path, path / "words.json")
        if trimmed:
            words = [word for word in words if word["start"] < MAX_SECONDS]

        original_cues = decode_srt((path / "original.srt").read_bytes())
        if trimmed:
            kept = []
            for cue in original_cues:
                if cue.start >= MAX_SECONDS:
                    continue
                if cue.end > MAX_SECONDS:
                    cue.end = MAX_SECONDS
                kept.append(cue)
            original_cues = kept

        state["words"] = words
        state["cues"] = [cue.to_dict() for cue in original_cues]
        state["status"] = "ready"
        state["progress"] = "Ready"
        state["error"] = None
        save_state(path, state)
    except Exception as exc:
        state["status"] = "error"
        state["progress"] = "Failed"
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes) and detail:
            message = detail.decode("utf-8", errors="replace")[-500:]
        else:
            message = str(exc) or traceback.format_exc()[-500:]
        state["error"] = message
        save_state(path, state)


def create_session(
    video_bytes: bytes,
    video_name: str,
    srt_bytes: bytes,
    glossary: list[str],
    background_tasks: BackgroundTasks | None = None,
    wait: bool = False,
) -> dict:
    suffix = Path(video_name).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO:
        raise HTTPException(
            status_code=400,
            detail="Upload an MP4 or WebM video (MOV and MKV also work).",
        )
    decode_srt(srt_bytes)  # validate early

    session_id = uuid.uuid4().hex
    path = DATA / session_id
    path.mkdir(parents=True, exist_ok=True)
    (path / f"upload{suffix}").write_bytes(video_bytes)
    (path / "original.srt").write_bytes(srt_bytes)
    state = {
        "id": session_id,
        "status": "processing",
        "progress": "Queued…",
        "glossary": glossary,
        "dismissed": [],
        "words": [],
        "cues": [],
        "error": None,
        "trimmed": False,
        "duration": None,
    }
    save_state(path, state)

    if wait or background_tasks is None:
        process_session(session_id)
        return public_state(path, load_state(path))

    background_tasks.add_task(process_session, session_id)
    return public_state(path, state)


def parse_glossary(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/sessions")
async def upload_session(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    srt: UploadFile = File(...),
    glossary: str = Form(""),
) -> dict:
    video_name = video.filename or "video.mp4"
    if Path(srt.filename or "").suffix.lower() != ".srt":
        raise HTTPException(status_code=400, detail="Upload one SRT caption file.")
    return create_session(
        await video.read(),
        video_name,
        await srt.read(),
        parse_glossary(glossary),
        background_tasks=background_tasks,
        wait=False,
    )


@app.post("/api/sessions/demo")
def demo_session(background_tasks: BackgroundTasks) -> dict:
    video = SAMPLES / "demo.mp4"
    srt = SAMPLES / "demo.srt"
    if not video.exists() or not srt.exists():
        raise HTTPException(status_code=404, detail="The planted sample is not in the samples folder.")
    return create_session(
        video.read_bytes(),
        video.name,
        srt.read_bytes(),
        ["mitosis"],
        background_tasks=background_tasks,
        wait=False,
    )


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    path = session_dir(session_id)
    return public_state(path, load_state(path))


@app.get("/api/sessions/{session_id}/video")
def session_video(session_id: str) -> FileResponse:
    path = session_dir(session_id)
    video = path / "video.mp4"
    if not video.exists():
        raise HTTPException(status_code=404, detail="Video is still processing or missing")
    return FileResponse(video, media_type="video/mp4")


@app.post("/api/sessions/{session_id}/cues/{cue_id}")
def edit_cue(session_id: str, cue_id: int, body: TextBody) -> dict:
    text = normalize_caption_text(body.text)
    if not text:
        raise HTTPException(status_code=400, detail="Caption text cannot be empty.")
    path = session_dir(session_id)
    state = load_state(path)
    if state.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Session is still processing")
    for cue in state["cues"]:
        if cue["id"] == cue_id:
            cue["text"] = text
            break
    else:
        raise HTTPException(status_code=404, detail="Unknown cue")
    save_state(path, state)
    return public_state(path, state)


@app.post("/api/sessions/{session_id}/dismiss")
def dismiss_flag(session_id: str, body: FlagBody) -> dict:
    path = session_dir(session_id)
    state = load_state(path)
    if state.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Session is still processing")
    flags = build_flags(
        cues_from_state(state),
        state["words"],
        state.get("glossary") or [],
        set(state.get("dismissed") or []),
    )
    if body.flagId not in {flag["id"] for flag in flags}:
        raise HTTPException(status_code=404, detail="Unknown possible issue")
    if body.flagId not in state["dismissed"]:
        state["dismissed"].append(body.flagId)
    save_state(path, state)
    return public_state(path, state)


@app.post("/api/sessions/{session_id}/apply")
def apply_flag(session_id: str, body: FlagBody) -> dict:
    text = normalize_caption_text(body.text or "")
    if not text:
        raise HTTPException(status_code=400, detail="Enter the caption text before applying it.")
    path = session_dir(session_id)
    state = load_state(path)
    if state.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Session is still processing")
    flags = build_flags(
        cues_from_state(state),
        state["words"],
        state.get("glossary") or [],
        set(state.get("dismissed") or []),
    )
    flag = next((item for item in flags if item["id"] == body.flagId), None)
    if flag is None:
        raise HTTPException(status_code=404, detail="Unknown possible issue")
    if flag["cueId"] is None:
        next_id = max((cue["id"] for cue in state["cues"]), default=0) + 1
        state["cues"].append(
            {"id": next_id, "start": flag["start"], "end": flag["end"], "text": text}
        )
    else:
        for cue in state["cues"]:
            if cue["id"] == flag["cueId"]:
                cue["text"] = text
                break
    save_state(path, state)
    return public_state(path, state)


@app.get("/api/sessions/{session_id}/original.srt")
def download_original(session_id: str) -> Response:
    path = session_dir(session_id)
    payload = (path / "original.srt").read_bytes()
    return Response(
        payload,
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="original.srt"'},
    )


@app.get("/api/sessions/{session_id}/reviewed.srt")
def download_reviewed(session_id: str) -> Response:
    path = session_dir(session_id)
    state = load_state(path)
    if state.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Session is still processing")
    payload = write_srt(cues_from_state(state)).encode("utf-8")
    (path / "reviewed.srt").write_bytes(payload)
    return Response(
        payload,
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="reviewed.srt"'},
    )
