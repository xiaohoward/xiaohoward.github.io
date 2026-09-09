"""Build conceptual.html (interactive) and conceptual.md (flat) from questions/*.md.

    python build.py            # writes conceptual.html + conceptual.md next to this file

Source format: see questions/_FORMAT.md. The parser is strict and fails loudly on malformed files.
"""
from __future__ import annotations

import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
QDIR = HERE / "questions"

Q_RE = re.compile(r"^## Q:\s*(?P<q>.+?)\s*\{diff=(?P<diff>[123])\s+tags=(?P<tags>[a-z0-9_,\-]+)\}\s*$")
W_RE = re.compile(r"^<!--\s*weight:\s*(\d)\s*-->\s*$")


@dataclass
class Question:
    id: str
    topic: str
    topic_idx: int
    question: str
    context_md: str
    answer_md: str
    diff: int
    tags: list[str]


@dataclass
class Topic:
    idx: int
    file: str
    title: str
    weight: int = 3
    questions: list[Question] = field(default_factory=list)


# ----------------------------------------------------------------------------- markdown -> html
def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    return s


MATH_TOKEN = "MATHSPAN{}Z"


def _protect_math(md: str) -> tuple[str, list[str]]:
    """Replace $$...$$ and $...$ spans (outside fenced code) with placeholders so the Markdown pass
    leaves them untouched; KaTeX auto-render finds the restored `$` delimiters in the page."""
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(0))
        return MATH_TOKEN.format(len(spans) - 1)

    parts = re.split(r"(```.*?```)", md, flags=re.S)
    for i in range(0, len(parts), 2):  # even indices are outside fences
        seg = parts[i]
        seg = re.sub(r"\$\$.+?\$\$", stash, seg, flags=re.S)
        seg = re.sub(r"(?<!\\)\$(?!\s)(?:[^$\n]|\\\$)+?(?<!\s)\$", stash, seg)
        parts[i] = seg
    return "".join(parts), spans


def _restore_math(html_out: str, spans: list[str]) -> str:
    def put(m: re.Match) -> str:
        return html.escape(spans[int(m.group(1))], quote=False)

    out = re.sub(r"MATHSPAN(\d+)Z", put, html_out)
    # a paragraph that is only a display block becomes a block-level math container
    out = re.sub(r"<p>(\$\$.*?\$\$)</p>", r'<div class="mathblock">\1</div>', out, flags=re.S)
    return out


def md_to_html(md: str) -> str:
    """Small deterministic Markdown subset: headings, paragraphs, bold/italic/code, fenced code,
    bullet and numbered lists (one nesting level), simple pipe tables, blockquotes. Math ($...$, $$...$$)
    is passed through untouched for KaTeX."""
    md, spans = _protect_math(md)
    return _restore_math(_md_to_html(md), spans)


def _md_to_html(md: str) -> str:
    out: list[str] = []
    lines = md.strip("\n").splitlines()
    i = 0
    para: list[str] = []

    def flush_para():
        if para:
            out.append("<p>" + _inline(" ".join(x.strip() for x in para)) + "</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            flush_para()
            lang = ln[3:].strip()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = f' class="lang-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush_para()
            lvl = min(len(m.group(1)) + 2, 6)  # demote so it sits under card headings
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if re.match(r"^\s*[-*+]\s+", ln) or re.match(r"^\s*\d+[.)]\s+", ln):
            flush_para()
            i = _list(lines, i, out, 0)
            continue
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-{2,}", lines[i + 1]):
            flush_para()
            i = _table(lines, i, out)
            continue
        if ln.startswith(">"):
            flush_para()
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i][1:].strip())
                i += 1
            out.append("<blockquote>" + _inline(" ".join(buf)) + "</blockquote>")
            continue
        if not ln.strip():
            flush_para()
            i += 1
            continue
        para.append(ln)
        i += 1
    flush_para()
    return "\n".join(out)


def _list(lines: list[str], i: int, out: list[str], depth: int) -> int:
    ordered = bool(re.match(r"^\s*\d+[.)]\s+", lines[i]))
    tag = "ol" if ordered else "ul"
    base_indent = len(lines[i]) - len(lines[i].lstrip())
    out.append(f"<{tag}>")
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            # blank line ends list only if the next non-blank line is not a list item at >= base indent
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and re.match(r"^\s*([-*+]|\d+[.)])\s+", lines[j]) and (
                len(lines[j]) - len(lines[j].lstrip())
            ) >= base_indent:
                i = j
                continue
            break
        indent = len(ln) - len(ln.lstrip())
        m = re.match(r"^\s*([-*+]|\d+[.)])\s+(.*)$", ln)
        if m and indent == base_indent:
            item = [m.group(2)]
            i += 1
            # continuation lines / nested list
            while i < len(lines):
                nxt = lines[i]
                nind = len(nxt) - len(nxt.lstrip())
                if nxt.strip() and re.match(r"^\s*([-*+]|\d+[.)])\s+", nxt) and nind > base_indent:
                    out.append("<li>" + _inline(" ".join(item)))
                    i = _list(lines, i, out, depth + 1)
                    out.append("</li>")
                    item = None
                    break
                if nxt.strip() and not re.match(r"^\s*([-*+]|\d+[.)])\s+", nxt) and nind > base_indent:
                    item.append(nxt.strip())
                    i += 1
                    continue
                break
            if item is not None:
                out.append("<li>" + _inline(" ".join(item)) + "</li>")
            continue
        if m and indent < base_indent:
            break
        if m and indent > base_indent:
            i = _list(lines, i, out, depth + 1)
            continue
        break
    out.append(f"</{tag}>")
    return i


def _table(lines: list[str], i: int, out: list[str]) -> int:
    def cells(s: str) -> list[str]:
        s = s.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [c.strip() for c in s.split("|")]

    head = cells(lines[i])
    i += 2
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        rows.append(cells(lines[i]))
        i += 1
    out.append('<div class="tbl"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return i


# ----------------------------------------------------------------------------- parsing
def parse_file(path: Path, idx: int) -> Topic:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = None
    weight = 3
    topic = None
    cur: dict | None = None
    mode = None  # "ctx" | "ans"
    qn = 0
    rapid = False  # questions after a '<!-- rapid-fire -->' marker get the 'rapid-fire' tag

    def close():
        nonlocal cur
        if cur is None:
            return
        if not cur["answer"].strip():
            raise SystemExit(f"{path.name}: question has empty answer: {cur['q'][:60]}")
        topic.questions.append(
            Question(
                id=f"{path.stem.split('_')[0]}-{len(topic.questions) + 1:02d}",
                topic=topic.title,
                topic_idx=idx,
                question=cur["q"],
                context_md=cur["ctx"].strip(),
                answer_md=cur["answer"].strip(),
                diff=cur["diff"],
                tags=cur["tags"],
            )
        )
        cur = None

    for n, ln in enumerate(lines, 1):
        if title is None:
            if ln.startswith("# "):
                title = ln[2:].strip()
                topic = Topic(idx=idx, file=path.name, title=title)
            elif ln.strip():
                raise SystemExit(f"{path.name}:{n}: file must start with '# Title'")
            continue
        wm = W_RE.match(ln)
        if wm and cur is None:
            topic.weight = int(wm.group(1))
            continue
        if re.match(r"^<!--.*-->\s*$", ln):
            if "rapid-fire" in ln:
                rapid = True
            continue
        if ln.startswith("## "):
            m = Q_RE.match(ln)
            if not m:
                raise SystemExit(f"{path.name}:{n}: bad question line (need '## Q: ... {{diff=N tags=a,b}}'):\n{ln}")
            close()
            qn += 1
            tags = m.group("tags").split(",")
            if rapid and "rapid-fire" not in tags:
                tags.append("rapid-fire")
            cur = {"q": m.group("q"), "diff": int(m.group("diff")), "tags": tags, "ctx": "", "answer": ""}
            mode = "ctx"
            continue
        if ln.strip() == "### A":
            if cur is None:
                raise SystemExit(f"{path.name}:{n}: '### A' before any question")
            if mode == "ans":
                raise SystemExit(f"{path.name}:{n}: duplicate '### A'")
            mode = "ans"
            continue
        if cur is not None:
            cur["ctx" if mode == "ctx" else "answer"] += ln + "\n"
        elif ln.strip():
            raise SystemExit(f"{path.name}:{n}: stray text before first question: {ln[:60]}")
    close()
    if topic is None or not topic.questions:
        raise SystemExit(f"{path.name}: no questions parsed")
    return topic


def load_topics() -> list[Topic]:
    files = sorted(p for p in QDIR.glob("[0-9][0-9]_*.md"))
    if not files:
        raise SystemExit(f"no question files in {QDIR}")
    return [parse_file(p, i) for i, p in enumerate(files)]


# ----------------------------------------------------------------------------- outputs
def write_flat_md(topics: list[Topic], out: Path) -> None:
    parts = ["# Interview prep — conceptual question bank\n"]
    for t in topics:
        parts.append(f"\n## {t.title}\n")
        for q in t.questions:
            parts.append(f"\n### [{q.id}] {q.question}  (difficulty {q.diff}; tags: {', '.join(q.tags)})\n")
            if q.context_md:
                parts.append(q.context_md + "\n")
            parts.append("\n<details><summary>Answer</summary>\n\n" + q.answer_md + "\n\n</details>\n")
    out.write_text("\n".join(parts), encoding="utf-8")


def build_html(topics: list[Topic]) -> str:
    data = {
        "topics": [{"idx": t.idx, "title": t.title, "weight": t.weight, "n": len(t.questions)} for t in topics],
        "questions": [
            {
                "id": q.id,
                "t": q.topic_idx,
                "q": _inline(q.question),
                "qtext": q.question,
                "ctx": md_to_html(q.context_md) if q.context_md else "",
                "a": md_to_html(q.answer_md),
                "d": q.diff,
                "tags": q.tags,
                "search": (q.question + " " + q.context_md + " " + q.answer_md + " " + " ".join(q.tags)).lower(),
            }
            for t in topics
            for q in t.questions
        ],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    template = (HERE / "template.html").read_text(encoding="utf-8")
    for marker in ("/*__DATA__*/", "<!--__KATEX__-->"):
        if marker not in template:
            raise SystemExit(f"template.html is missing the {marker} marker")
    seed_path = HERE / "progress.json"
    seed = seed_path.read_text(encoding="utf-8").strip() if seed_path.exists() else "null"
    json.loads(seed)  # must be valid JSON
    return (
        template.replace("<!--__KATEX__-->", katex_assets())
        .replace("/*__DATA__*/", "const DATA = " + payload + ";")
        .replace("/*__SEED__*/", "const SEED = " + seed.replace("</", "<\\/") + ";")
    )


def katex_assets() -> str:
    """Self-contained KaTeX: CSS with the woff2 fonts embedded as data URIs, plus katex + auto-render JS.
    Assets live in vendor/ (KaTeX 0.16.11 from cdnjs); the page then needs no network and works in the Artifact
    sandbox, which blocks external stylesheets and fonts."""
    import base64

    vend = HERE / "vendor"
    css = (vend / "katex.min.css").read_text(encoding="utf-8")

    def font(m: re.Match) -> str:
        data = base64.b64encode((vend / m.group(1)).read_bytes()).decode()
        return f"url(data:font/woff2;base64,{data})"

    # keep only the woff2 sources (drop woff/ttf fallbacks, which would be broken relative paths)
    css = re.sub(r"url\((fonts/[A-Za-z0-9_-]+\.woff2)\)", font, css)
    css = re.sub(r",url\(fonts/[A-Za-z0-9_-]+\.(?:woff|ttf)\) format\(\"(?:woff|truetype)\"\)", "", css)
    js = (vend / "katex.min.js").read_text(encoding="utf-8")
    auto = (vend / "auto-render.min.js").read_text(encoding="utf-8")
    return f"<style>{css}</style>\n<script>{js}</script>\n<script>{auto}</script>"


def main() -> None:
    topics = load_topics()
    n = sum(len(t.questions) for t in topics)
    (HERE / "conceptual.html").write_text(build_html(topics), encoding="utf-8")
    write_flat_md(topics, HERE / "conceptual.md")
    for t in topics:
        print(f"  {t.file:38s} {len(t.questions):3d} q  weight {t.weight}")
    print(f"built conceptual.html + conceptual.md: {len(topics)} topics, {n} questions")
    if n < 100:
        print("WARNING: fewer than 100 questions", file=sys.stderr)


if __name__ == "__main__":
    main()
