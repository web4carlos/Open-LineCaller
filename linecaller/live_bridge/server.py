from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .officiating_csv import CsvOfficiatingTimeline
from .state import LiveCourtState, LiveStateStore
from .video import VideoSourceRegistry, content_type_for, parse_range_header


def demo_store():
    s = LiveStateStore("Open LineCaller Demo Club")
    s.upsert(LiveCourtState(
        court_id="court-1",
        name="Center Court",
        status="LIVE",
        mode="MATCH",
        score="7 - 5 - 1",
        last_call="SCANNING",
        confidence=None,
        nearest_line="",
        distance_in=None,
        evidence="",
        serving_team="TEAM A",
        server_number=1,
        service_court="RIGHT COURT",
        camera_id="CAM-01",
        session_id="MATCH-001",
    ))
    return s


def handler_for(store, videos, timelines, *, hold_frames=45):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header(
                "Access-Control-Allow-Origin",
                "http://127.0.0.1:5173",
            )
            self.send_header("Cache-Control", "no-store")

        def _json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _stream_video(self, court_id):
            source = videos.get(court_id)
            if source is None or not source.path.is_file():
                return self._json(
                    404,
                    {"error": "video_not_configured", "court_id": court_id},
                )

            path = source.path
            size = path.stat().st_size
            requested = self.headers.get("Range")
            try:
                byte_range = parse_range_header(requested, size)
            except (TypeError, ValueError):
                byte_range = None

            if requested and byte_range is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self._cors()
                self.end_headers()
                return

            start, end = byte_range if byte_range else (0, size - 1)
            length = end - start + 1

            self.send_response(206 if byte_range else 200)
            self.send_header("Content-Type", content_type_for(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if byte_range:
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{end}/{size}",
                )
            self.send_header(
                "Access-Control-Allow-Origin",
                "http://127.0.0.1:5173",
            )
            self.end_headers()

            try:
                with path.open("rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining:
                        chunk = f.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path == "/api/health":
                return self._json(
                    200,
                    {"ok": True, "service": "open-linecaller-live-bridge"},
                )

            if path == "/api/live":
                payload = store.snapshot().to_dict()
                for court in payload["courts"]:
                    court["video"] = videos.describe(court["court_id"])
                    timeline = timelines.get(court["court_id"])
                    court["officiating"] = {
                        "available": timeline is not None,
                        "event_count": len(timeline.events) if timeline else 0,
                        "source": "FINAL_OFFICIATING_CSV" if timeline else None,
                    }
                return self._json(200, payload)

            if path.startswith("/api/video/"):
                return self._stream_video(path.split("/")[-1])

            if path.startswith("/api/officiating/"):
                court_id = path.split("/")[-1]
                timeline = timelines.get(court_id)
                if timeline is None:
                    return self._json(
                        404,
                        {"error": "officiating_events_not_configured"},
                    )

                source = videos.get(court_id)
                payload = timeline.to_dict()
                payload.update({
                    "court_id": court_id,
                    "hold_frames": hold_frames,
                    "fps": source.fps if source else 60.0,
                    "video_width": source.width if source else None,
                    "video_height": source.height if source else None,
                    "source": "FINAL_OFFICIATING_CSV",
                })
                return self._json(200, payload)

            self._json(404, {"error": "not_found"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--video")
    p.add_argument("--officiating-csv")
    p.add_argument("--court-id", default="court-1")
    p.add_argument("--hold-frames", type=int, default=45)
    a = p.parse_args()

    videos = VideoSourceRegistry()
    timelines = {}

    if a.video:
        videos.bind(a.court_id, a.video)

    if a.officiating_csv:
        timelines[a.court_id] = CsvOfficiatingTimeline.load(
            a.officiating_csv
        )

    server = ThreadingHTTPServer(
        (a.host, a.port),
        handler_for(
            demo_store(),
            videos,
            timelines,
            hold_frames=a.hold_frames,
        ),
    )

    print(f"Open LineCaller Live Bridge: http://{a.host}:{a.port}")
    print(
        f"Video: {a.court_id} -> {Path(a.video).resolve()}"
        if a.video else
        "Video: not configured"
    )
    print(
        f"Officiating CSV: {a.court_id} -> "
        f"{Path(a.officiating_csv).resolve()}"
        if a.officiating_csv else
        "Officiating CSV: not configured"
    )
    if a.officiating_csv:
        print(
            f"Final officiating events loaded: "
            f"{len(timelines[a.court_id].events)}"
        )
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
