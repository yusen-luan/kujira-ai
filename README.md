# Autonomous ML Research Agent — KuaiRand-Pure

TikTok TechJam 2026 submission (Track 2). An LLM-driven agent that autonomously runs
the ML engineering loop — inspect data, engineer features, train and tune, evaluate, reflect
and iterate — on the KuaiRand-Pure recommendation benchmark, using only train + validation
feedback, until it converges. For the full design writeup (why
the agent is built this way, and a detailed account of what it tried), see
[`docs/SOLUTION.md`](docs/SOLUTION.md).

## Results

| Split | Metric | Official baseline | This agent | Δ |
|---|---|---|---|---|
| Validation | primary (GAUC+nDCG@5 avg) | 0.6016 | **0.6039** | +0.0023 |
| Test (organizer-designated hidden-test date range) | primary | 0.5946 | **0.5975** | +0.0029 |

Converged after 19 iterations (1 accepted: a BPR pairwise-ranking training loss, found via a
literature-cited web-research call) — full per-metric breakdown, the accept/reject history,
and resource usage in [`docs/SOLUTION.md`](docs/SOLUTION.md#5-results).

## Setup and installation

- **Python 3.10+** and the one dependency in `requirements.txt` (numpy — the whole pipeline
  is numpy + stdlib only):
  ```
  pip install -r requirements.txt
  ```
- **The KuaiRand-Pure data files** (organizer-provided, not redistributed in this repo — too
  large). Place these four CSVs in `workspace/data/`:
  ```
  workspace/data/log_standard_4_08_to_4_21_pure.csv
  workspace/data/log_standard_4_22_to_5_08_pure.csv
  workspace/data/user_features_pure.csv
  workspace/data/video_features_basic_pure.csv
  ```
- **Claude Code CLI** (only needed if you want to re-run the autonomous agent itself, see
  §3 below — not needed to reproduce the results in the table above). Install per
  [docs.claude.com/claude-code](https://docs.claude.com/claude-code), then authenticate with
  **one** of:
  - `claude login` once, interactively, on the machine you'll run this on (uses a Claude
    subscription session — this is how the original run was done), **or**
  - `export ANTHROPIC_API_KEY=sk-ant-...` in the shell before running (direct API billing,
    no login step — the CLI picks this up automatically in headless `-p` mode too).

  If a flag in a command below comes back as "unrecognized," update the CLI
  (`npm install -g @anthropic-ai/claude-code`) — this was built against a recent version.

## Steps to reproduce results

**1. Reproduce the accepted checkpoint's exact metrics (deterministic, no LLM call, ~1 min):**
```
python make_submission.py --split valid
```
This retrains `agent/runs/best/baseline.py` (the accepted BPR-loss FM — see
[`docs/SOLUTION.md`](docs/SOLUTION.md#4-what-actually-happened-in-the-run)) with the exact
seed and hyperparameters recorded in `logs/node_4.json`, and prints valid/test primary scores.
You should see `valid primary 0.6035 | test primary 0.5975`, matching the table above.

**2. Generate and validate the final submission CSV:**
```
python make_submission.py                                          # writes submission/submission.csv (test split)
cd workspace && python submit.py ../submission/submission.csv --check --score --data_dir ./data
```
(On Windows, if the checker's checkmark output crashes with a `UnicodeEncodeError`, prefix
the second command with `PYTHONIOENCODING=utf-8` — a console-encoding quirk in the starter
kit's `submit.py`, unrelated to the submission content itself.)

**3. (Optional) Re-run the autonomous agent from scratch.** This is non-deterministic (LLM
proposals vary run to run) and costs real API spend (~$3.58 for the 18-iteration run behind
the results above) — use it to watch the actual autonomy in action, not to reproduce an exact
number:
```
python agent/orchestrator.py --no-resume --iterations 15
```
`--no-resume` archives the current `agent/runs/`/`logs/` state first (nothing is deleted) and
starts a fresh chain from the organizer's pristine baseline. Drop `--no-resume` to instead
continue iterating on top of the current best. See `python agent/orchestrator.py --help` for
budget/timeout/model flags.

## Project structure

```
workspace/            starter kit, as provided: data.py (loading+splits+encoding),
                       baseline.py (pop/random/FM/BPR-FM), evaluate.py (PINNED scoring —
                       GAUC + nDCG@5 -> primary), submit.py (writes/validates submission.csv)
                       models/  reference PyTorch model zoo (DeepFM/DCN/AutoInt/xDeepFM),
                                 not currently wired into the pipeline (see docs/SOLUTION.md §6)

agent/                 the autonomous agent
  orchestrator.py       propose -> apply -> run -> evaluate -> accept/reject loop,
                         error recovery, convergence check
  llm.py                headless Claude Code CLI wrapper (two roles: no-tools propose/repair,
                         and a separate WebSearch/WebFetch-only research role)
  eda.py, rag.py         one-time EDA pass + literature retrieval, feeding every proposal
  web_research.py        live, domain-allowlisted web research (a third grounding source)
  literature/            9 curated, source-verified method notes (checked in)
  prompt_templates.py    system prompt + the LEVER_CATEGORY taxonomy
  runs/                  best/ (current accepted pipeline code) + node_N/ (one working copy
                          per iteration) + eda/literature outputs — all checked in

logs/node_N.json        per-iteration record: hypothesis, lever category, before/after
                         metrics, hyperparameter sweep, error/recovery attempts
submission/             final submission CSV(s)
make_submission.py      generates submission/submission.csv from the current best checkpoint
docs/SOLUTION.md         full design writeup + results + limitations (this file's companion)
```

## Limitations and what I'd improve with more time

The run converged per the organizer's rule (ε=0.002, N=3) after covering every one of the
agent's 9 lever categories at least once — the plateau reflects a genuinely exhausted search
of incremental changes to a numpy FM, not premature stopping. The highest-expected-value next
step is wiring in one of the already-present but unused PyTorch model architectures
(`workspace/models/`) as a structurally different model family, since everything tried so far
was a variation on the same FM scaffolding. Token consumption (input+output) isn't separately
logged, only per-call dollar cost — a fixable gap in `agent/llm.py`, not a missing capability.
Full detail on both, plus every rejected hypothesis and why: [`docs/SOLUTION.md`](docs/SOLUTION.md#6-limitations-and-what-id-improve-with-more-time).

## Team

Yusen Luan (solo).
