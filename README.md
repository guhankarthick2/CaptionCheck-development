# CaptionCheck

CaptionCheck shows a teacher which captions might change the meaning of a lesson, so they can listen, fix those lines, and export a corrected SRT.

It is a **local web app**. Upload a short English video and an existing `.srt` file (optional glossary). CaptionCheck transcribes the audio independently, queues possible issues (numbers, negation, glossary terms, speech with no caption), and lets a person correct or dismiss them. The original caption file is never overwritten.

## Requirements

- [Python 3.11+](https://www.python.org/downloads/) (on Windows, check **Add Python to PATH**)
- [FFmpeg](https://ffmpeg.org/download.html) on PATH  
  Windows (PowerShell): `winget install --id Gyan.FFmpeg -e`
- After installing FFmpeg, **open a new terminal** so `ffmpeg` is recognized

## Launch

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Leave that terminal open. In a browser, open:

**http://127.0.0.1:8000**

Click **Open planted sample** to load the built-in demo clip, or upload your own MP4/WebM and SRT.

The first run downloads the `small.en` speech model and can take several minutes. Later runs reuse the cache.

Clips longer than 90 seconds are trimmed. Every row is a **possible issue**; CaptionCheck does not silently rewrite numbers or negations.

## Stop the server

In the terminal that is running uvicorn, press `Ctrl+C`.
