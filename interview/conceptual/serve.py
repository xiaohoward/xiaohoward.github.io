"""Study with progress synced to the git repo.

    python serve.py            # git pull, rebuild, serve on http://localhost:8765/conceptual.html, commit progress on change
    python serve.py --no-git   # just rebuild and serve (no pull/commit)
    python serve.py --push     # also git push after each commit
    python serve.py --pages    # on exit, mirror the kit + progress to https://xiaohoward.github.io/interview/

While the page is open from this server, every verdict / rating / note / edit is written to conceptual/progress.json
and committed (debounced). Ctrl+C stops the server (and pushes once if --push). On another machine: git pull, then
run this again — the page merges the pulled progress.json with whatever that browser already had, newest wins.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROGRESS = HERE / "progress.json"


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=REPO, check=check, capture_output=True, text=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def log_message(self, fmt, *args):  # quieter
        if "progress.json" in (args[0] if args else ""):
            return
        super().log_message(fmt, *args)

    def do_POST(self):
        if self.path != "/progress.json":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        data = json.loads(body)  # must be valid
        PROGRESS.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        self.server.dirty_at = time.time()
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def committer(server, use_git, push):
    """Commit progress.json when it has been quiet for 20 s."""
    last_commit = 0.0
    while not server.stop:
        time.sleep(2)
        if server.dirty_at and time.time() - server.dirty_at > 20 and server.dirty_at > last_commit:
            last_commit = time.time()
            if use_git:
                git("add", str(PROGRESS.relative_to(REPO)))
                r = git("commit", "-m", "progress: study session", check=False)
                if r.returncode == 0:
                    print("  committed progress.json")
                    if push:
                        git("push", check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--pages", action="store_true", help="on exit, also mirror to xiaohoward.github.io/interview via ../publish_pages.py")
    a = ap.parse_args()
    use_git = not a.no_git
    if use_git:
        r = git("pull", "--ff-only", check=False)
        print("git pull:", (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "ok")
    subprocess.run([sys.executable, str(HERE / "build.py")], check=True, cwd=HERE)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    server.dirty_at = 0.0
    server.stop = False
    threading.Thread(target=committer, args=(server, use_git, a.push), daemon=True).start()
    url = f"http://localhost:{a.port}/conceptual.html"
    print(f"serving {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.stop = True
    if use_git:
        git("add", str(PROGRESS.relative_to(REPO)))
        r = git("commit", "-m", "progress: study session", check=False)
        if r.returncode == 0:
            print("committed progress.json")
        if a.push:
            print(git("push", check=False).stderr.strip() or "pushed")
    if a.pages:
        subprocess.run([sys.executable, str(REPO / "publish_pages.py")], cwd=REPO)


if __name__ == "__main__":
    main()
