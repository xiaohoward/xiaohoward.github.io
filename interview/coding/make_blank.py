"""Generate blank (candidate) versions of coding problems from the reference solutions.

For every <tier>/solution/NN_name.py, reads `__implement__` (a list of "func" or "Class.method"
names) and rewrites those bodies as a TODO stub that keeps the signature and docstring:

    def foo(x):
        \"\"\"docstring kept\"\"\"
        # TODO: implement
        raise NotImplementedError

Everything else (helpers, scaffolds, imports, the interview prompt docstring) is copied verbatim.

Usage:  python make_blank.py            # all tiers
        python make_blank.py advanced   # one tier
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIERS = ("basic", "advanced")


def _find_targets(tree: ast.Module, names: list[str]) -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = f"{node.name}.{sub.name}"
                    if key in names:
                        found[key] = sub
    missing = set(names) - set(found)
    if missing:
        raise SystemExit(f"__implement__ names not found as top-level def / Class.method: {sorted(missing)}")
    return found


def blank_source(src: str) -> str:
    tree = ast.parse(src)
    names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__implement__" for t in node.targets
        ):
            names = ast.literal_eval(node.value)
    if not names:
        raise SystemExit("solution has no `__implement__ = [...]` list")

    targets = _find_targets(tree, names)
    lines = src.splitlines(keepends=True)
    # Replace bodies from the bottom up so earlier line numbers stay valid.
    for fn in sorted(targets.values(), key=lambda f: f.lineno, reverse=True):
        body = fn.body
        has_doc = (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )
        if not has_doc:
            raise SystemExit(f"{fn.name}: every __implement__ target must have a docstring (it is the spec)")
        first_stmt = body[1] if len(body) > 1 else None
        # Lines to cut: from the first statement after the docstring to the function end.
        start = (first_stmt.lineno if first_stmt else body[0].end_lineno + 1) - 1
        end = fn.end_lineno  # exclusive index in 0-based list
        indent = " " * (body[0].col_offset)
        stub = [
            f"{indent}# TODO(candidate): implement. Read the docstring above; run the tests with ../run.py\n",
            f"{indent}raise NotImplementedError\n",
        ]
        lines[start:end] = stub
    out = "".join(lines)
    ast.parse(out)  # must still be valid Python
    return out


def main(argv: list[str]) -> None:
    tiers = argv or list(TIERS)
    for tier in tiers:
        sol_dir = HERE / tier / "solution"
        blank_dir = HERE / tier / "blank"
        blank_dir.mkdir(parents=True, exist_ok=True)
        for sol in sorted(sol_dir.glob("[0-9][0-9]_*.py")):
            out = blank_dir / sol.name
            out.write_text(blank_source(sol.read_text()))
            print(f"wrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main(sys.argv[1:])
