"""Derive conceptual_artifact.html (wrapper-free copy for publishing as a claude.ai Artifact) from conceptual.html.

The Artifact host supplies its own <html>/<head>/<body>, blocks file downloads, and allows Google Fonts, so this
copy: strips the document wrappers, adds the IBM Plex fonts, and swaps file export/import for copy-paste modals.
Run after build.py:  python make_artifact.py   then republish the file at the same artifact URL.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
h = (HERE / "conceptual.html").read_text(encoding="utf-8")
h = re.sub(r"^<!doctype html>\s*<html[^>]*>\s*<head>\s*", "", h, flags=re.S | re.I)
h = re.sub(r"<meta[^>]*>\s*", "", h)
h = h.replace("</head>\n<body>\n", "").replace("</body>\n</html>\n", "").replace("</body>\n</html>", "")
fonts = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
h = h.replace("<title>Interview Prep — Conceptual Drills</title>", "<title>Conceptual Drills</title>\n" + fonts)
h = h.replace("font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif",
              "font:15px/1.55 'IBM Plex Sans',system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif")
mono = "font-family:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;"
h = h.replace("code{background:var(--code);", "code{" + mono + "background:var(--code);")
h = h.replace("pre{background:var(--code);", "pre{" + mono + "background:var(--code);")

old_btns = '''      <button class="small" id="btnExport">Export progress</button>
      <label class="small" style="cursor:pointer"><input type="file" id="importFile" accept="application/json" class="hidden"><button class="small" type="button" onclick="document.getElementById('importFile').click()">Import</button></label>'''
assert old_btns in h, "template changed: update make_artifact.py"
h = h.replace(old_btns, '''      <button class="small" id="btnExport">Export progress</button>
      <button class="small" id="btnImport">Import</button>''')
start = h.index("document.getElementById('btnExport').onclick")
end = h.index("document.getElementById('btnReset').onclick")
h = h[:start] + r'''document.getElementById('btnExport').onclick = () => {
  const txt = JSON.stringify({exported:new Date().toISOString(), state});
  openModal(`<h2 style="margin-top:0">Export progress</h2><p class="sub">Copy this JSON and keep it somewhere (notes, a file). Paste it back with Import on any device.</p>
    <textarea id="exportBox" class="note" style="min-height:160px;font-family:'IBM Plex Mono',monospace;font-size:12px" readonly>${esc(txt)}</textarea>
    <div class="row"><button class="primary" id="copyBtn">Copy to clipboard</button><button onclick="closeModal()">Close</button></div>`);
  const box = document.getElementById('exportBox'); box.select();
  document.getElementById('copyBtn').onclick = () => { box.select(); try { navigator.clipboard.writeText(txt).then(()=>{document.getElementById('copyBtn').textContent='Copied';}); } catch(e){ document.execCommand('copy'); } };
};
document.getElementById('btnImport').onclick = () => {
  openModal(`<h2 style="margin-top:0">Import progress</h2><p class="sub">Paste the JSON from a previous export. This replaces the progress stored in this browser.</p>
    <textarea id="importBox" class="note" style="min-height:160px;font-family:'IBM Plex Mono',monospace;font-size:12px" placeholder="{...}"></textarea>
    <div class="row"><button class="primary" id="importGo">Import</button><button onclick="closeModal()">Cancel</button></div>`);
  document.getElementById('importGo').onclick = () => {
    try { const j = JSON.parse(document.getElementById('importBox').value); state = j.state || j; save(); closeModal(); render(); }
    catch(err){ alert('That is not valid export JSON.'); }
  };
};
''' + h[end:]
assert h.startswith("<title>")
(HERE / "conceptual_artifact.html").write_text(h, encoding="utf-8")
print("wrote conceptual_artifact.html", len(h) // 1024, "KB")
