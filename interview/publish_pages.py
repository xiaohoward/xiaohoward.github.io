"""Mirror the kit into the public GitHub Pages site so it is reachable from anywhere.

    python publish_pages.py                 # rebuild, copy into <pages repo>/interview/, commit, push
    python publish_pages.py --pages-repo /path/to/xiaohoward.github.io

What goes up: everything in this repo except .git, caches, the artifact copy, and the job-description files
(1.txt, 2.txt, 4.txt). The built conceptual.html (with progress.json embedded, so your study history is
reflected) is also written as interview/index.html, so the page is https://xiaohoward.github.io/interview/ .
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXCLUDE_TOP = {".git", "1.txt", "2.txt", "4.txt"}
EXCLUDE_ANY = {"__pycache__", "conceptual_artifact.html", ".DS_Store"}


def copy_tree(src: Path, dst: Path) -> int:
    n = 0
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if rel.parts[0] in EXCLUDE_TOP or any(part in EXCLUDE_ANY for part in rel.parts):
            continue
        if p.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        else:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages-repo", default=str(HERE.parent / "xiaohoward.github.io"))
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    pages = Path(a.pages_repo).resolve()
    if not (pages / ".git").exists():
        raise SystemExit(f"{pages} is not a git checkout; clone xiaohoward/xiaohoward.github.io there first")
    subprocess.run([sys.executable, str(HERE / "conceptual" / "build.py")], check=True, cwd=HERE / "conceptual")
    target = pages / "interview"
    target.mkdir(exist_ok=True)
    # drop files that no longer exist upstream (keeps the mirror exact); the folder is fully regenerated
    for p in list(target.iterdir()):
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    n = copy_tree(HERE, target)
    shutil.copy2(HERE / "conceptual" / "conceptual.html", target / "index.html")
    (pages / ".nojekyll").touch()  # Jekyll would otherwise skip files starting with '_'
    print(f"copied {n} files into {target}")
    subprocess.run(["git", "add", "-A", "interview", ".nojekyll"], cwd=pages, check=True)
    r = subprocess.run(["git", "commit", "-q", "-m", "interview: sync prep kit + study progress"], cwd=pages)
    if r.returncode != 0:
        print("nothing to commit")
        return
    if not a.no_push:
        subprocess.run(["git", "push"], cwd=pages, check=True)
        print("pushed: https://xiaohoward.github.io/interview/")


if __name__ == "__main__":
    main()
