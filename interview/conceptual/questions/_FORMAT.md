# Authoring format for conceptual question files (parsed by ../build.py)

File = one topic. Structure (exact markers matter):

```
# <Topic Title>
<!-- weight: 3 -->          <- integer 1-5, sampling weight in mock-interview mode (3 = normal, 5 = core)

## Q: <one-line question, as an interviewer would say it> {diff=2 tags=diffusion,sampling}
Optional context or interviewer follow-ups, as short bullets (shown WITH the question, before reveal):
- Follow-up: ...
### A
<answer in markdown>
```

Rules
- `## Q:` line MUST end with `{diff=<1|2|3> tags=<comma-separated,no spaces>}`. diff 1 = warm-up, 2 = standard,
  3 = deep/advanced. Always include at least one tag; use lowercase.
- `### A` starts the answer. Everything until the next `## Q:` (or EOF) is the answer.
- Answer style: what a strong candidate would SAY. Lead with the direct answer in 1-2 sentences, then the key
  equation / mechanism / number, then the common trap or follow-up. 80-250 words typical; deep ones up to 350.
  Use **bold** for the key phrase. Use bullet lists and fenced code blocks where helpful.
- Math: LaTeX, rendered by KaTeX. Inline `$x_t = (1-t)\,x_0 + t\,\epsilon$`; display `$$ ... $$` on its own
  lines (use `\begin{aligned} ... \end{aligned}` inside `$$` for multi-line derivations). Keep prose outside the
  dollars. A literal dollar sign in prose must be written `\$`. Never put math inside fenced code blocks unless it
  is code. Use `\text{...}` for words inside math and `\mathrm{d}` for differentials.
- No HTML tags. No images. Avoid tables wider than ~4 columns.
- Questions must be answerable verbally. Include "why", "what breaks", "how would you scale" style questions.
- Do NOT number questions; the build assigns ids from file order.
