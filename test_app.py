import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as app_module
from app import app


class ReviewFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.patch = patch.object(app_module, "DATA", self.data)
        self.patch.start()
        self.client = TestClient(app)
        self.words = [
            {"start": 0.2, "end": 0.5, "word": "Do", "probability": 0.95},
            {"start": 0.5, "end": 0.8, "word": "not", "probability": 0.96},
            {"start": 0.8, "end": 1.1, "word": "mix", "probability": 0.94},
            {"start": 1.1, "end": 1.6, "word": "these", "probability": 0.9},
            {"start": 1.6, "end": 2.2, "word": "chemicals", "probability": 0.92},
            {"start": 4.0, "end": 4.3, "word": "Use", "probability": 0.95},
            {"start": 4.3, "end": 4.6, "word": "zero", "probability": 0.9},
            {"start": 4.6, "end": 4.9, "word": "point", "probability": 0.9},
            {"start": 4.9, "end": 5.2, "word": "zero", "probability": 0.9},
            {"start": 5.2, "end": 5.6, "word": "five", "probability": 0.93},
            {"start": 5.6, "end": 6.2, "word": "milligrams", "probability": 0.9},
            {"start": 8.0, "end": 8.4, "word": "Rinse", "probability": 0.9},
            {"start": 8.4, "end": 8.7, "word": "with", "probability": 0.9},
            {"start": 8.7, "end": 9.2, "word": "water", "probability": 0.91},
            {"start": 9.2, "end": 9.6, "word": "right", "probability": 0.9},
            {"start": 9.6, "end": 10.1, "word": "away", "probability": 0.9},
        ]

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def wait_ready(self, session_id: str) -> dict:
        for _ in range(50):
            payload = self.client.get(f"/api/sessions/{session_id}").json()
            if payload.get("status") in {"ready", "error"}:
                return payload
            time.sleep(0.05)
        self.fail("session never became ready")

    def upload(self):
        srt = (
            "1\n00:00:00,100 --> 00:00:02,500\nMix these chemicals.\n\n"
            "2\n00:00:03,800 --> 00:00:06,500\nUse 0.5 milligrams.\n"
        ).encode()
        with (
            patch.object(app_module, "media_duration_seconds", return_value=12.0),
            patch.object(app_module, "extract_wav", lambda source, wav: wav.write_bytes(b"RIFFstub")),
            patch.object(app_module, "load_or_transcribe", lambda wav, dest: self.words),
        ):
            response = self.client.post(
                "/api/sessions",
                files={
                    "video": ("lesson.mp4", b"fake-video", "video/mp4"),
                    "srt": ("lesson.srt", io.BytesIO(srt), "application/x-subrip"),
                },
                data={"glossary": "mitosis"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        if payload.get("status") != "ready":
            payload = self.wait_ready(payload["id"])
        self.assertEqual(payload["status"], "ready", payload)
        return payload, srt

    def test_flags_edit_dismiss_and_preserve_original(self):
        payload, srt = self.upload()
        kinds = {flag["kind"] for flag in payload["flags"]}
        self.assertEqual(kinds, {"negation", "number", "missing"})
        session_id = payload["id"]
        original = self.client.get(f"/api/sessions/{session_id}/original.srt")
        self.assertEqual(original.content, srt)

        number = next(flag for flag in payload["flags"] if flag["kind"] == "number")
        applied = self.client.post(
            f"/api/sessions/{session_id}/apply",
            json={"flagId": number["id"], "text": "Use 0.05 milligrams."},
        )
        self.assertEqual(applied.status_code, 200)
        self.assertNotIn("number", {flag["kind"] for flag in applied.json()["flags"]})

        missing = next(flag for flag in applied.json()["flags"] if flag["kind"] == "missing")
        added = self.client.post(
            f"/api/sessions/{session_id}/apply",
            json={"flagId": missing["id"], "text": "Rinse with water right away."},
        )
        self.assertEqual(added.status_code, 200)
        self.assertNotIn("missing", {flag["kind"] for flag in added.json()["flags"]})
        self.assertTrue(any("Rinse" in cue["text"] for cue in added.json()["cues"]))

        negation = next(flag for flag in added.json()["flags"] if flag["kind"] == "negation")
        dismissed = self.client.post(
            f"/api/sessions/{session_id}/dismiss",
            json={"flagId": negation["id"]},
        )
        self.assertEqual(dismissed.json()["flags"], [])

        reviewed = self.client.get(f"/api/sessions/{session_id}/reviewed.srt")
        self.assertIn(b"Use 0.05 milligrams.", reviewed.content)
        self.assertIn(b"Rinse with water right away.", reviewed.content)
        self.assertEqual(self.client.get(f"/api/sessions/{session_id}/original.srt").content, srt)
        self.assertNotEqual(reviewed.content, srt)


if __name__ == "__main__":
    unittest.main()
