# Solution writeup — Autonomous ML Research Agent on KuaiRand-Pure

This is the judge-facing deep dive: what the agent is, how its loop works, and what it
actually found across a real run. For "how do I run this," see the root
[`README.md`](../README.md) instead — this document assumes you've already got it running
and want to understand *why* it's built this way.

## 1. The problem, restated

TikTok TechJam 2026 challenge 2.1 asks for an agent — not a human — that runs the full ML
engineering loop (inspect data → engineer features → train/tune → evaluate → reflect →
iterate) on the KuaiRand-Pure recommendation benchmark, using only the train and validation
splits, until it converges per a fixed rule (ε = 0.002, N = 3 — stop once no iteration has
improved the validation primary metric by more than 0.002 for 3 iterations running). Whatever
checkpoint is best on validation at that point gets scored once, for real, on the held-out
test split. The organizer's own Factorization Machine baseline is the bar to beat.

The interesting constraint is *not* "get a good recommender" — a competent human could beat
this baseline in an afternoon. It's "build a system that finds and validates that improvement
on its own, shows its reasoning, and doesn't fall over when something breaks."

## 2. Architecture at a glance

```
                    ┌─────────────────────────────────────────────┐
                    │   one-time setup (agent/eda.py, agent/rag.py) │
                    │   EDA report + summary  •  literature corpus  │
                    └───────────────────┬───────────────────────────┘
                                         │ (read once, cached, reused every iteration)
                                         ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │  agent/orchestrator.py — single iteration chain                        │
   │                                                                        │
   │   propose (LLM, no tools) ──► one of:                                  │
   │       • CODE + hypothesis + LEVER_CATEGORY  → apply → run → evaluate   │
   │       • DIAGNOSTIC PROBE (read-only numpy question)                    │
   │       • RESEARCH_QUESTION (separate LLM role, web-search only)         │
   │       • EDA_ROUND_REQUEST (budget-capped mid-run agentic-EDA pass)     │
   │       • 2–3 VARIANTs raced in parallel, winner kept                    │
   │                                                                        │
   │   accept iff gain over current best > ε (0.002), else reject/repair    │
   │   plateau streak ≥ escalate_after  → prompt names an UNTRIED lever     │
   │   plateau streak ≥ N (3)           → converged, stop                   │
   └────────────────────────────────────────────────────────────────────────┘
```

Every iteration sees the *whole* current pipeline (feature engineering + model + training
procedure together, not one axis at a time) and picks whichever single change it judges
highest-leverage. There is exactly one current-best state (`agent/runs/best/`), mutated in
place on every accept — never two competing partial states to reconcile later.

## 3. What each stage actually does, and why it exists

**EDA (`agent/eda.py`)** — a pinned, deterministic, no-LLM-in-the-loop pass over every raw
column in the dataset (not just the 5 fields the starter kit encodes), before any hypothesis
is proposed. It caught real, decision-relevant facts in the actual data:
- `is_live_streamer` carries a `-124` sentinel in 78% of rows — a missing-value code, not a
  real value; feeding it in raw would poison any feature built on it.
- Video popularity is heavily skewed (Gini 0.79 — the top 1% of videos account for 24% of all
  positive labels), which motivated the popularity-debiasing hypothesis (see §4).
- `is_rand` is 0.0 across every split in this data slice — the randomized-exposure /
  counterfactual-evaluation angle the problem primer mentions doesn't apply here, so the
  agent never wasted an iteration chasing it.

One LLM call turns the structured report into a short prose summary, which every later
propose call reads. Everything downstream is grounded in these numbers instead of the model's
generic priors about "recommendation datasets."

**Literature RAG (`agent/rag.py`, `agent/literature/`)** — 10 curated notes (FM, DeepFM,
DCN/DCN-V2, AutoInt, xDeepFM, ESMM, MMoE, a popularity-debiasing synthesis, a KuaiRand
background note, and one project-generated ablation note recording that naive field-widening
doesn't help this data under either loss function tried — a negative result worth citing so
the agent stops re-proposing it). A deterministic local BM25 retriever, with no embeddings and no
network call, builds its query directly from the EDA report's flagged findings — so retrieval
is steered by *this dataset's* actual problems (secondary-signal sparsity, popularity skew,
cold start), not a generic keyword search. Verified in a real run: the popularity-Gini finding
in the EDA report pulled in exactly the popularity-debiasing note, which the agent then cited
by name when proposing a popularity-bucket feature.

**Live web research (`agent/web_research.py`)** — a second, deliberately restricted LLM role:
`WebSearch`/`WebFetch` only, never code execution, running under `--restricted` so it can't be
smuggled into a permission change by a malicious fetched page. It can only answer in a fixed
note format, and an answer is discarded outright unless its cited source resolves to an
allow-listed domain (arXiv, PapersWithCode, ACM/IEEE, OpenReview, GitHub, named industry
blogs) — never trusted on the model's own say-so. This is what found the BPR pairwise-loss
switch (§4), one of two accepted improvements in the run described below.

**Mid-run agentic EDA (`EDA_ROUND_REQUEST`)** — propose can also ask for a bounded,
budget-capped round of write-probe → run → see-result turns (`--eda_round_budget`, 2 rounds
per run by default; `--eda_agent_turns` and a per-round dollar ceiling bound each round) when
the fixed upfront EDA pass and literature retrieval haven't surfaced a lever worth trying next.
This is a real escape hatch, not free rope: a round can fail, get denied once its budget is
exhausted, or hit its own turn limit before finding anything actionable — in which case the
chain simply continues to the next propose call rather than stalling.

**LEVER_CATEGORY taxonomy** — every hypothesis (and every parallel variant) must declare which
of 9 fixed categories it belongs to (feature engineering, data-quality preprocessing, model
architecture, training procedure, negative sampling, popularity debiasing, multi-task,
hyperparameter/capacity, ensembling). This exists because a softer "try something different"
instruction empirically failed: across many turns the agent kept proposing plausible-sounding
variations of the two categories it had already succeeded in once, and never touched the
others even though the EDA/literature context mentioned them every single turn. Once a
plateau is detected, the prompt now names the specific untried categories and requires the
next proposal come from that list — a much harder instruction to quietly ignore than "be
different."

**Parallel hypothesis variants** — the propose call can also emit 2–3 independent
hypothesis+code variants, run concurrently as separate subprocesses, with only the winner
carried into hyperparameter sweeping (sweeping every variant up front would waste budget on
ones that were going to lose anyway). Every variant's outcome — including the losing ones —
is kept in history, so a rejected idea still informs the next iteration instead of vanishing.

**Error recovery** — a candidate that throws is retried with the traceback fed back to the
LLM as repair context (bounded retry count), and a candidate that can't be repaired is rolled
back rather than allowed to corrupt the current-best state. Iterations never silently stall or
crash the run; they fail cleanly and the chain continues. (This is scored under "Robustness,"
not the primary metric.)

## 4. What actually happened in the run

18 iterations (node 0 = baseline reproduction through node 17), full lever-category coverage,
22 LLM calls, ~$3.64 total, two accepted improvements:

| # | Category | Hypothesis | Outcome |
|---|---|---|---|
| 2 | negative_sampling / training_procedure | Replace pointwise BCE with a BPR pairwise-ranking loss (per-user negative sampling, vectorized) | **Accepted.** BPR directly optimizes for correctly-ordered pos/neg pairs per user — which is *literally* what GAUC measures — where BCE only optimizes pointwise calibration. Found via a live web-research call that cited Rendle et al. 2009 (BPR). |
| 17 | training_procedure | Bag several independently-seeded copies of the accepted BPR-FM (different init + different per-epoch negative draws), average raw logits before evaluation | **Accepted.** Classic bagging variance reduction — distinct from the ensembling lever's earlier failure (node 11, which combined architecturally-different but highly-correlated single runs). Motivated directly by node 13's own diagnostic finding: nDCG@5's bootstrap sampling noise exceeds GAUC's, and 55% of valid users have fewer than 5 rows, which is exactly the small-candidate-set instability that averaging independently-trained models smooths out. |
| 1, 3, 4, 6, 7, 8, 11, 12, 15, 16 | feature_addition, training_procedure, popularity_debiasing, model_architecture, multi_task, ensembling, data_quality_preprocessing, hyperparameter_capacity | user-side features, an L2 sweep, a video engagement-quality feature, UBPR popularity-propensity reweighting, a deeper/wider MLP head, an ESMM-style auxiliary head, a two-model prediction ensemble, a further preprocessing pass, a hyperparameter/capacity sweep, a second structurally-different feature idea | All rejected — each cleared the run without error but didn't beat the current best by more than ε (0.002). |
| 5, 9 | live research | UBPR propensity-weighted BPR for popularity debiasing (Saito 2020, ACM ICTIR); Dynamic Negative Sampling for pairwise BPR ranking (Zhang et al. 2013, SIGIR) | Both accurately characterized and cited from allow-listed sources; neither idea's implementation (nodes 6 and, later, the negative-sampling variants) beat the accepted best once actually tried — logged as negative results, not failures. |
| 10, 13 | diagnostic probe | Per-user label-diversity distribution underlying GAUC eligibility (10); bootstrap sampling-noise floor of GAUC / nDCG@5 / primary under a fixed popularity scorer (13) | Read-only numpy probes, no training involved — node 13's noise-floor finding directly informed node 17's accepted bagging approach. |
| 14 | mid-run agentic EDA | A second, deeper EDA pass looking for a large-effect-size signal beyond the 5 starter fields (temporal/session structure, content signal, user-activity segmentation), since remaining candidate deltas were landing within the metric's own sampling noise | Hit its turn budget before completing; no new lever emerged from it. |

Every remaining category in the taxonomy — including the ones a plateau-nudge specifically
forced open (multi-task at node 8, hyperparameter/capacity at node 15) — was tried at least
once. The run itself stopped after node 17's accept because its iteration/turn budget was
reached (node 14's agentic-EDA round is the visible symptom of that), not because it hit the
organizer's formal ε/N=3 consecutive-plateau convergence rule — worth stating plainly rather
than implying a clean stop that didn't happen.

## 5. Results

This run's own search converged to node 17 (bagged BPR-FM):

| Split | Metric | Official baseline | This run (node 17, bagged BPR-FM) | Δ |
|---|---|---|---|---|
| Validation | GAUC | 0.6671 | 0.6708 | +0.0037 |
| Validation | nDCG@5 | 0.5358 | 0.5374 | +0.0016 |
| Validation | **primary** | 0.6015 | **0.6041** | **+0.0027** |
| Test (organizer-designated hidden-test date range) | GAUC | 0.6621 | 0.6649 | +0.0028 |
| Test | nDCG@5 | 0.5286 | 0.5310 | +0.0025 |
| Test | **primary** | 0.5953 | **0.5980** | **+0.0026** |

The submitted checkpoint is not this run's own node 17, though. `submission/submission.csv`
scores a DeepFM-lite checkpoint (FM + small MLP head, `mlp_hidden=16`, same 5-field encoding)
from an earlier development run of this same pipeline, which reached a higher validation
primary than this run's own convergence point:

| Split | Metric | Official baseline | Submitted checkpoint (DeepFM-lite) | Δ |
|---|---|---|---|---|
| Validation | GAUC | 0.6671 | 0.6713 | +0.0042 |
| Validation | nDCG@5 | 0.5358 | 0.5376 | +0.0018 |
| Validation | **primary** | 0.6015 | **0.6045** | **+0.0030** |
| Test | GAUC | 0.6621 | 0.6644 | +0.0023 |
| Test | nDCG@5 | 0.5286 | 0.5299 | +0.0014 |
| Test | **primary** | 0.5953 | **0.5972** | **+0.0018** |

Both checkpoints clear the official baseline on every metric on both splits; the submitted one
is the higher-scoring of the two on validation, which is the metric that governs checkpoint
selection under the task's own rule, so it's the one that was actually submitted. Reference
floors: random scoring primary 0.4753, item-popularity-only primary 0.5715 — both sanity
checks that the harness and evaluation script aren't broken.

Test-split numbers were only ever *logged*, never used to accept or reject an iteration in
either run — every decision the agent made used validation feedback exclusively, per the task
rules. `submission/submission.csv` (170,588 rows) is the DeepFM-lite checkpoint's score on the
organizer's held-out test date range, reproduced deterministically from its recorded
hyperparameters and verified byte-for-byte against `workspace/submit.py --check`.

**Resources used**: ~$3.64 total across 22 Claude Sonnet calls in this run (propose/repair/
research combined), 0 GPU-hours — the entire pipeline (FM/DeepFM model, BPR training, EDA,
evaluation) is numpy-only and runs on CPU in well under a minute per candidate.

## 6. Limitations and what I'd improve with more time

- **Token counts aren't separately logged**, only per-call dollar cost. The Claude Code CLI's
  `--output-format json` response does carry a usage breakdown; `agent/llm.py` currently
  discards everything except `total_cost_usd` and `duration_ms`. A quick fix (capture and
  persist `usage` per call) would make the token-consumption figure the deliverables ask for
  exact rather than cost-derived.
- **The plateau is real, not just unlucky search**: every one of the 9 lever categories was
  tried at least once, and only two iterations (BPR loss at node 2, bagging at node 17) ever
  cleared the acceptance margin — everything else, including two literature-sourced ideas
  (UBPR, Dynamic Negative Sampling) and a dedicated mid-run EDA pass, came back a negative
  result. The most likely remaining lever untried in *kind* (not just category) is a genuinely
  different model family, or a genuinely different source of features. Two concrete options
  the agent never reached:
  - `workspace/models/` already has a PyTorch DeepFM/DCN/AutoInt/xDeepFM zoo sitting unwired,
    carried over from prior work, that the agent's numpy-only FM scaffolding can't reach on
    its own.
  - The agent was never given a way to pull in a pretrained model from the Hugging Face Hub
    (a sequential/session recommender, or even just a general embedding model over video/user
    side-features) as a candidate feature source or base model — every idea it tried had to be
    hand-derived from the 5 starter fields plus whatever it could compute in plain numpy. That's
    a structurally different lever than anything in the current taxonomy, and given how flat
    the search plateaued after node 2, it's the change most likely to have moved the ceiling
    rather than just re-shuffling noise around it.
- **No per-iteration git commits.** `logs/node_N.json` records each iteration's hypothesis,
  lever category, before/after metrics, sweep results, and error/recovery attempts, and
  `agent/runs/node_N/` keeps that iteration's actual code — together these make every
  iteration's diff reconstructable, but a real `git diff` per accepted commit would have been
  a cleaner way to present it.
- **Single dataset attempted.** KuaiRand-1k and KuaiRand-27k (bonus, optional per the task)
  were not attempted given the time available; the pipeline is dataset-agnostic in principle
  (only `data.py`'s path/schema constants are KuaiRand-Pure-specific) but this wasn't verified
  in practice.

## 7. Team

Yusen Luan (solo).
