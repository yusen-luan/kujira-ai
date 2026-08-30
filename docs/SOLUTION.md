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

**Literature RAG (`agent/rag.py`, `agent/literature/`)** — 9 curated, source-verified notes
(FM, DeepFM, DCN/DCN-V2, AutoInt, xDeepFM, ESMM, MMoE, a popularity-debiasing synthesis, and a
KuaiRand background note). A deterministic local BM25 retriever, with no embeddings and no
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
blogs) — never trusted on the model's own say-so. This is what found the one change that
actually worked (§4).

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

19 iterations (node 0 = baseline reproduction through node 18), full lever-category coverage,
24 LLM calls, ~$3.95 total, one accepted improvement:

| # | Category | Hypothesis | Outcome |
|---|---|---|---|
| 4 | training_procedure | Replace pointwise BCE with a BPR pairwise-ranking loss (per-user negative sampling, vectorized) | **Accepted.** BPR directly optimizes for correctly-ordered pos/neg pairs per user — which is *literally* what GAUC measures — where BCE only optimizes pointwise calibration. Found via a live web-research call that cited Rendle et al. 2009 (BPR). |
| 5–18 | feature_addition, data_quality_preprocessing, model_architecture, negative_sampling, popularity_debiasing, multi_task, hyperparameter_capacity, ensembling | popularity-bucket feature, recency-weighting, dynamic negative sampling, field-weighted FM, multi-task auxiliary head, inverse-popularity reweighting, two rounds of 2-model prediction ensembling, hyperparameter sweeps | All rejected — each cleared the run without error but didn't beat the current best by more than ε. Two live-research calls (Dynamic Negative Sampling, FwFM) were accurately characterized but didn't pan out on this specific dataset; logged as negative results, not failures. Node 18's ensemble (BPR-FM + a `tag`-feature variant) got closest (0.6042, +0.0007 short of the bar). |

The run converged per the organizer's own rule: 14 consecutive iterations covering every
remaining lever category without a >0.002 gain. That is the intended stopping condition, not
a symptom of the agent giving up early — every category in the taxonomy was tried at least
once (including the three the plateau-nudge specifically forced open: multi-task,
recency-weighting, and popularity-reweighting) before the run called itself converged.

## 5. Results

| Split | Metric | Official baseline | This run (node 4, BPR-loss FM) | Δ |
|---|---|---|---|---|
| Validation | GAUC | 0.6674 | 0.6702 | +0.0028 |
| Validation | nDCG@5 | 0.5357 | 0.5369 | +0.0012 |
| Validation | **primary** | 0.6016 | **0.6035** | **+0.0019** |
| Test (organizer-designated hidden-test date range) | GAUC | 0.6610 | 0.6644 | +0.0034 |
| Test | nDCG@5 | 0.5282 | 0.5307 | +0.0025 |
| Test | **primary** | 0.5946 | **0.5975** | **+0.0029** |

Reference floors: random scoring primary 0.4753, item-popularity-only primary 0.5715 — both
sanity checks that the harness and evaluation script aren't broken.

The test-split numbers above were only ever *logged*, never used to accept or reject an
iteration — every decision the agent made used validation feedback exclusively, per the
task rules. `submission/submission.csv` (170,588 rows) is this checkpoint's score on that same
test date range, generated by `make_submission.py` and verified byte-for-byte against
`workspace/submit.py --check`/`--score`.

**Resources used**: ~$3.58 total across 23 Claude Sonnet calls (propose/repair/research
combined), 0 GPU-hours — the entire pipeline (FM model, BPR training, EDA, evaluation) is
numpy-only and runs on CPU in well under a minute per candidate.

## 6. Limitations and what I'd improve with more time

- **Token counts aren't separately logged**, only per-call dollar cost. The Claude Code CLI's
  `--output-format json` response does carry a usage breakdown; `agent/llm.py` currently
  discards everything except `total_cost_usd` and `duration_ms`. A quick fix (capture and
  persist `usage` per call) would make the token-consumption figure the deliverables ask for
  exact rather than cost-derived.
- **The plateau is real, not just unlucky search**: every one of the 9 lever categories was
  tried at least once and none cleared the acceptance margin after node 4. The most likely
  remaining lever untried in *kind* (not just category) is a genuinely different model family
  — `workspace/models/` already has a PyTorch DeepFM/DCN/AutoInt/xDeepFM zoo sitting unwired,
  carried over from prior work, that the agent's numpy-only FM scaffolding can't reach on its
  own. Wiring one of those in as a candidate model family is the highest-expected-value next
  experiment, precisely because it's a structurally different lever than anything tried so
  far, not an incremental nudge on the same one.
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
