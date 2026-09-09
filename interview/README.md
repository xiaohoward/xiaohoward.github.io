# Interview prep kit — NVIDIA PhD research internships (GenAI / CV&DL / Robotics) and similar

Built 2026-09-04 from the Sept 2026 CV (`cv/`), the three NVIDIA job descriptions (`1.txt`, `2.txt`, `4.txt`), and
the candidate brief in `_context_brief.md` (what interviewers will grill: SPEED, foveated imaging/diffusion,
single-photon ToF, KV-cache gating for world-action models, distributed training claims).

```
conceptual/            click-to-reveal question bank (100+ questions)
  conceptual.html      OPEN THIS in a browser (double-click; works offline from file://)
  conceptual.md        same bank as flat Markdown with <details> answers (GitHub/VS Code preview)
  questions/*.md       the source — edit / add questions here, then rebuild
  build.py             python build.py  -> regenerates conceptual.html + conceptual.md
  vendor/              KaTeX 0.16.11 (js, css, woff2 fonts) inlined by build.py
  progress.json        THE progress file (schedule, ratings, notes, edits) — tracked in git
  serve.py             python serve.py -> pull, build, serve locally, write+commit progress.json on every change
  make_artifact.py     python make_artifact.py -> conceptual_artifact.html, the copy published at
                       https://claude.ai/code/artifact/d14ecc55-a436-40a9-b6a1-7ed96618bf2e (phone/anywhere)
coding/                interview coding drills, two tiers
  run.py               the test runner (see below)
  basic/               building blocks from scratch: 01-14 classic ML/DL, 15-24 CV & generative,
                       25-36 genAI / JD fundamentals (schedules, CFG, samplers, VQ, CLIP, LM loss, video tokens,
                       VLA actions, BC & diffusion policy, physics integrators, image conditioning) (36)
  advanced/            diffusion / flow / DiT / systems / your-own-research problems (14)
    blank/             what you get in the interview: prompt + signatures + docstrings, bodies are TODO
    solution/          answer key (don't open until you've tried)
    tests/             shared tests, run against either
  make_blank.py        regenerates blank/ from solution/ (blank never drifts from the key)
```

## Environment
Use the `speed` conda env (numpy + torch CPU are enough):
```
conda activate speed
```

## Conceptual drills

### Progress lives in the repo: `conceptual/progress.json`
Verdict history, intervals/ease, ratings, notes and edited answers are all in that one file. The daily way to study:
```
cd conceptual && python serve.py        # git pull → rebuild → opens http://localhost:8765/conceptual.html
# ... study ...                          # every change is written to progress.json and committed (debounced)
# Ctrl+C                                # add --push to push automatically, otherwise: git push
```
On another machine: `git pull`, then `python serve.py` again. The page merges the pulled file with whatever that
browser already stored, newest record per question wins, so nothing is lost in either direction.

**Public copy on the web:** `python publish_pages.py` (or `python serve.py --pages`, which does it on exit) mirrors
this kit into the `interview/` folder of the `xiaohoward.github.io` checkout next to this repo and pushes, so
https://xiaohoward.github.io/interview/ always shows the page with your history as of the last push. The job
description files are never copied there. The public page is read-only for progress: what you do in that browser
stays in that browser (copy-and-paste Export / Import to bring it back).

Opening `conceptual.html` directly from the file system still works (build.py embeds progress.json into the page
as a seed, and the browser keeps its own copy), but then changes stay in that browser until you either run
serve.py or paste an Export into progress.json by hand. The published artifact is a separate copy: its state stays
in that browser; use its copy-and-paste Export / Import to move progress.

- **Flashcards** is the default view: one question at a time. Reveal (`space`), grade (`1`/`2`/`3`, which
  auto-advances), rate difficulty, edit or annotate, then `j`/`→` next and `k`/`←` back. All filters, modes and
  Shuffle apply to the deck. Switch to **List** for the scrollable overview with Reveal all / Hide all.
- **Verdict** after revealing: "Did you answer correctly?" → Correct (`1`) / Partly (`2`) / Incorrect (`3`).
  This drives an SM-2 scheduler (the algorithm behind Anki/SuperMemo): each card has an ease factor (starts 2.5)
  and an interval. Correct → 1 day, then 6 days, then interval × ease (ease +0.1 each time); Partly → a shorter
  interval and ease −0.14; Incorrect → relearn in 4 hours, interval reset, ease −0.54 (floor 1.3). Cards under
  60% lifetime accuracy come back sooner. Each card shows its interval, ease, next due, and predicted recall from
  an exponential forgetting curve.
- **Review session** (sidebar button): walks everything due plus N new cards (default 20). Anything marked
  Incorrect is re-inserted a few cards later and repeats until you get it; the session ends with a summary. This
  is the daily loop; the Dashboard shows the due forecast for today / 3 days / 7 days and average predicted recall.
- **Rate difficulty** ("Hard for me" 1–5) per question. The rating scales the intervals (1 → ×2, 2 → ×1.4, 3 → ×1,
  4 → ×0.6, 5 → ×0.35) and how often mock mode picks the question. Click the same number again to clear it.
- **Edit answers** in place ("Edit answer" → type → "Save"); "Revert to original" restores the built-in answer.
  Edits are stored with your progress (browser storage + export JSON), not in the Markdown sources.
- **Modes**: All / Drill weak (shaky+missed) / Due (spaced review) / Unseen / Rapid-fire (short definitional
  "what is X" questions; every topic has a block of these at the end, plus the dedicated file 14).
- **Math** is LaTeX rendered by KaTeX (bundled into the page with its fonts, so it works offline and in the
  artifact). Write `$...$` inline and `$$...$$` for display math in the question sources; the build compiles cleanly
  and a validator can check every span with KaTeX before publishing.
- **Filters**: topic, difficulty (1 warm-up, 2 standard, 3 deep), tags, free-text search (`/`).
- **Mock interview**: N questions sampled by topic weight (diffusion, image/video, distributed training and the
  SPEED grill are weight 5) and biased to your weak items; per-question timer; answer out loud, reveal, grade;
  summary table at the end.
- **Dashboard**: per-topic mastery bars, weakest tags, everything you missed last time.
- **My notes** box under each answer: what to say better next time, mnemonics, follow-ups you were asked; saved with your progress.

Suggested loop (45-60 min/day): **Review session** (answer out loud before revealing, be honest with the verdict)
→ one `Mock interview` of 6-8 questions with a 3-minute timer → one coding problem → notes on the weak ones.

### Adding or editing questions
Edit `conceptual/questions/NN_topic.md` (format in `questions/_FORMAT.md`), then:
```
cd conceptual && python build.py
```
The build fails loudly on malformed questions. Question ids are `<file-number>-<order>`; inserting a question
renumbers the ones after it in that file (their saved progress is keyed by id, so append rather than insert if you
want to keep history).

## Coding drills
```
cd coding
python run.py                                # list everything
python run.py basic 03 --show                # read the prompt as the interviewer would give it
#   ... implement the TODO stubs in basic/blank/03_kmeans.py in your editor ...
python run.py basic 03                       # test YOUR version (blank/)
python run.py basic 03 --against solution    # sanity-check the key
python run.py advanced --all                 # run all your advanced attempts
```
Each blank file starts with the interview prompt (problem statement, time budget, discussion follow-ups) and keeps
the full docstring of every function you must implement. Time yourself against the stated budget. After you pass,
read the solution and the "Discussion follow-ups" and rehearse those answers — the coding round almost always ends
with "what's the complexity / how does this scale / where is this used".

To start over on a problem, re-run `python make_blank.py basic` (it overwrites `blank/` from `solution/`, so copy
your attempt elsewhere first if you want to keep it).

## Coverage map (CV skills → where they are tested)
| CV claim | Conceptual topic files | Coding |
|---|---|---|
| Diffusion models, flow matching | 03 | basic 25, 26, 27, 28, 36; adv 01, 02, 03 |
| Image and video generation | 04, 10 | basic 15, 16, 18, 19, 22, 23; adv 11, 12 |
| Efficient deep learning, efficient attention, inference optimization | 05 | basic 11, adv 05, 14 |
| Long-context modeling, causal video generation | 07 | basic 31, 32; adv 05, 12, 13 |
| World models, VLA / world-action models, physical AI | 08 | basic 33, 34, 35; adv 12, 13 |
| Distributed and multi-GPU training | 06 | adv 08, 09, 10 |
| Model fine-tuning (LoRA) | 04, 11, 12 | adv 06, 07 |
| SPEED / Foveated imaging / Foveated diffusion (projects) | 10, 11 | adv 11 |
| Single-photon ToF, Bell LLM work, math rigor | 12 | basic 05, 06, 07, 30, 31 |
| Multimodal / VLM (JD), VQ tokenizers | 04, 08 | basic 29, 30 |
| PyTorch / Python | all | all |
| ML fundamentals, architectures | 01, 02 | basic 01-14, adv 04 |
| CV & 3D (JD2), robotics (JD4) | 09, 08 | basic 17, 20, 21, 24 |
| Research story / behavioral | 13 | — |
| Practical training / PyTorch rapid-fire (checkpointing, autocast, compile, OOM, resume) | 14 + rapid-fire blocks in every file | — |
