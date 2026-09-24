#!/usr/bin/env python3
"""Dev data server for the dashboard: serves eval/traces + answers on :5174
under /data/, and (re)generates eval/traces/index.json. Run from repo root:
    ./.venv/bin/python scripts/serve_ui_data.py
"""
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "eval" / "traces"
ANSWERS = ROOT / "answers"
DISCOVERY = ROOT / "eval" / "pattern_discovery.json"
DIST = ROOT / "ui" / "dist"


def build_index() -> None:
    TRACES.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.name[: -len(".trace.json")] for p in TRACES.glob("*.trace.json"))
    (TRACES / "index.json").write_text(json.dumps(ids))
    print(f"index.json: {len(ids)} traces")


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        if path.startswith("/data/traces/"):
            return str(TRACES / path[len("/data/traces/"):])
        if path.startswith("/data/answers/"):
            return str(ANSWERS / path[len("/data/answers/"):])
        if path == "/data/discovery.json":
            return str(DISCOVERY)
        if path == "/" or not path.startswith("/data/"):
            # serve the built dashboard (fallback to index.html for SPA routes)
            candidate = DIST / path.lstrip("/")
            if path == "/" or not candidate.is_file():
                return str(DIST / "index.html")
            return str(candidate)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args) -> None:  # quiet
        pass


def main() -> int:
    build_index()
    srv = ThreadingHTTPServer(("127.0.0.1", 5174), Handler)
    print("serving dashboard + /data on http://127.0.0.1:5174 (no-store)")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
