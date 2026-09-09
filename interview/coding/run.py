"""Interview coding drill runner.

    python run.py                          # list all problems in both tiers
    python run.py basic                    # list problems in a tier
    python run.py basic 03                 # run tests for basic/blank/03_*.py  (your attempt)
    python run.py basic 03 --against solution   # run the same tests against the reference
    python run.py advanced --all           # run every problem in a tier against blank
    python run.py advanced --all --against solution
    python run.py basic 03 --show          # print the interview prompt (module docstring) and stop

Workflow: open <tier>/blank/NN_name.py, read the prompt + docstrings, implement, then run this.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

# Tiny CPU tensors are dominated by per-op thread overhead on many-core machines: cap threads early.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

HERE = Path(__file__).resolve().parent
TIERS = ("basic", "advanced")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def problems(tier: str) -> list[Path]:
    return sorted((HERE / tier / "solution").glob("[0-9][0-9]_*.py"))


def find(tier: str, key: str) -> Path:
    for p in problems(tier):
        if p.name.startswith(key) or p.stem == key or p.stem[3:] == key:
            return p
    raise SystemExit(f"no problem matching '{key}' in {tier}. Try: python run.py {tier}")


def run_one(tier: str, sol_path: Path, against: str) -> bool:
    target = HERE / tier / against / sol_path.name
    test_path = HERE / tier / "tests" / f"test_{sol_path.name}"
    if not target.exists():
        raise SystemExit(f"missing {target} (run make_blank.py?)")
    print(f"\n=== {tier}/{against}/{sol_path.name} ===")
    t0 = time.perf_counter()
    try:
        mod = load(target, f"{tier}_{against}_{sol_path.stem}")
        tests = load(test_path, f"test_{tier}_{sol_path.stem}")
        tests.run(mod)
    except NotImplementedError:
        print("  ... NotImplementedError: you still have TODO stubs in this file")
        return False
    except AssertionError as e:
        print(f"  FAIL  {e}")
        tb = traceback.extract_tb(sys.exc_info()[2])
        for fr in tb:
            if "tests" in fr.filename:
                print(f"        at {Path(fr.filename).name}:{fr.lineno}  {fr.line}")
        return False
    except Exception:
        print("  ERROR")
        traceback.print_exc(limit=4)
        return False
    print(f"  PASS  ({time.perf_counter() - t0:.2f}s)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tier", nargs="?", choices=TIERS)
    ap.add_argument("problem", nargs="?", help="number (03) or name (kmeans)")
    ap.add_argument("--against", choices=("blank", "solution"), default="blank")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--show", action="store_true", help="print the prompt and exit")
    a = ap.parse_args()

    if a.tier is None:
        for tier in TIERS:
            print(f"{tier}:")
            for p in problems(tier):
                print(f"  {p.stem}")
        return
    if a.all:
        results = [run_one(a.tier, p, a.against) for p in problems(a.tier)]
        print(f"\n{sum(results)}/{len(results)} passed ({a.tier}, {a.against})")
        sys.exit(0 if all(results) else 1)
    if a.problem is None:
        for p in problems(a.tier):
            print(f"  {p.stem}")
        return
    p = find(a.tier, a.problem)
    if a.show:
        mod = load(HERE / a.tier / "blank" / p.name, "show")
        print(mod.__doc__)
        return
    ok = run_one(a.tier, p, a.against)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
