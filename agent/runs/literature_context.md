(retrieved via local BM25 search over agent/literature/ + agent/runs/web_literature/ + agent/runs/repo_notes/, query terms: CTR prediction ranking feature interaction recommendation embedding KuaiRand ablation field count vocab size capacity sparse imbalance multi-task auxiliary task sample selection bias seesaw popularity bias popularity skew long tail debias exposure bias gini cold start data quality sentinel value missing value)

### [curated] ESMM (Entire Space Multi-Task Model)  (X. Ma et al., "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate," SIGIR 2018)
**Core idea**: Built for a chain of dependent binary outcomes (impression -> click -> conversion). Two towers share one embedding table: a CTR-style tower predicts P(click | impression) over *every* impression; a CVR-style tower's raw output is never trained or scored on its own — it's only ever multiplied by the CTR tower's probability to directly estimate P(click AND next-event) over the whole impression space. This sidesteps two problems a naive "train a classifier only on clicked rows" approach hits: sample-selection bias (the downstream model would only ever see clicked examples during training, but must score every impression at inference) and extreme sparsity of the downstream positive label.

**When it helps**: exactly the pattern this project's own EDA flagged — `is_like`/`is_follow`/`is_comment`/`is_forward` are all under 3% positive and are logically downstream of engagement (unlikely to comment on something never really watched). Training a standalone classifier directly for one of these hits the same sample-selection-bias-plus-sparsity problem ESMM was designed for. `long_view` (this project's actual label) could play the role of the upstream "click" stage; a rarer signal like `is_like` or `is_comment` could play the downstream "conversion" stage.

**Cost**: two small MLP towers sharing one embedding table — roughly 2x a single-task model's cost, no separate downstream-labeled dataset needed since both stages are logged on the same rows here.

**KuaiRand-Pure notes**: no existing implementation in `workspace/models/*.py` — the zoo's variants are single-task or shared-bottom multi-task, not this specific sequential-multiplicative structure, so this would need new code rather than adapting an existing file.

### [curated] Popularity bias & debiasing  (survey synthesis: M. Klimashevskaia et al., "A Survey on Popularity Bias in Recommender Systems," arXiv 2308.01118 (2023); J. Chen et al., "Bias and Debias in Recommender System: A Survey and Future Directions," arXiv 2010.03240 (2020))
**Core idea**: Logged interaction data over-represents popular items because *exposure itself* is already biased by whatever recommender served the historical logs — popular items get shown, and hence get positive labels, far more often regardless of true per-user relevance. A model trained naively on this data partly relearns "what was popular" rather than "what's relevant." Standard mitigation families: (a) re-weight training examples inversely to item/exposure popularity, (b) regularize against or explicitly subtract a learned popularity-only signal from the main score, (c) causal / inverse-propensity-scoring (IPS) methods that reweight by an estimated exposure probability (need a source of unbiased/randomized exposure to estimate propensities from), (d) popularity-aware negative sampling.

**When it helps**: when a model's ranking looks like it's mostly rediscovering item popularity rather than adding personalization signal beyond it — this project's own EDA found exactly this pattern (Gini=0.79 on video-level positive counts, top 1% of videos = ~24% of all positives), and the `pop`-only baseline already scores uncomfortably close to the FM baseline (0.5715 vs 0.6016 primary) — evidence popularity alone already explains a large share of the achievable score here.

**Cost**: (a)/(d) are close to free — a training-loop reweighting or sampling change, no architecture change needed; (b)/(c) need an auxiliary popularity or propensity estimate but are still lightweight relative to a full model swap.

**KuaiRand-Pure notes**: IPS-style methods (c) specifically need the randomized-exposure log to estimate unbiased propensities — see `kuairand_dataset.md`: this project's downloaded data has `is_rand == 0.0` on every row, because the separate `log_random_*.csv` file wasn't part of what got downloaded. (a)/(b)/(d) have no such dependency and remain usable as-is.

### [found via live web search this run] Propensity-Weighted (Popularity-Based) BPR for Debiasing Popularity Skew — UBPR  (Saito 2020, ACM ICTIR ("Unbiased Pairwise Learning from Biased Implicit Feedback"))
Saito (ICTIR 2020) proposes Unbiased BPR (UBPR): instead of changing how negatives are sampled, each observed pairwise (positive-vs-unobserved) term in the BPR loss is reweighted by the inverse of an estimated exposure propensity for the positive item, where propensity is estimated purely from in-log item popularity raised to a tunable power θ (commonly θ≈0.5, following the standard power-law propensity estimator used in unbiased-recommendation literature, e.g. Schnabel et al. and Yang et al.), so no external exposure/click logs beyond item interaction counts are required. This is lightweight (one extra per-item weight array, O(1) lookup per gradient step) and is straightforward to implement with numpy alongside a standard BPR-FM training loop by simply multiplying the sigmoid-gradient term for each positive by 1/propensity(item). The official reference implementation is public at github.com/usaito/unbiased-pairwise-rec, confirming the method's simplicity and reproducibility. Note: search results also surfaced a caveat from a BPR replicability study (arXiv 2409.14217) that plain popularity-based *oversampling of negatives* (Rendle's original adaptive/context-dependent sampler) can underperform uniform sampling on full (non-sampled) ranking metrics — so for a KuaiRand-Pure-style sparse, popularity-skewed long_view task, propensity-reweighting the loss (UBPR-style) is the better-evidenced lightweight fix versus naively skewing negative sampling toward popular items; both approaches only need item interaction-frequency counts, not true exposure logs.

### [this project's own internal analysis] Naive field-widening does not improve FM on this data (ablation, both loss functions)  (this project's own ablation, workspace/ablation_features.py (verified 2026-08-31 under both the original pointwise-BCE loss and the current best's BPR pairwise loss))
Before proposing another single-field addition, know this: `workspace/ablation_features.py` already
tests whether widening the FM's encoded field set helps, by comparing the current 5-field baseline
(user_id, video_id, author_id, tab, dur_bucket) against +4 item-side fields (8 fields total: adds
music_id/video_type/upload_type) and the full CWM 13-field superset (+6 more user-side fields: follow/
fans/friend counts, register_days, user_active_degree — all as `_range`-bucketed raw categoricals, one
extra vocab slot per field). Verified twice, under both training objectives used in this project so far:

- Pointwise BCE (the original starter-kit loss): 5 fields → test primary 0.5950±0.0003; +4 item-side →
  0.5940±0.0004 (worse); full CWM-13 → 0.5940±0.0005 (worse).
- BPR pairwise loss (the current accepted best's mechanism): 5 fields → test primary 0.5968±0.0006;
  +4 item-side → 0.5968±0.0008 (statistically indistinguishable — no gain); full CWM-13 →
  0.5964±0.0009 (slightly worse, though within noise of the 5-field result).

Conclusion, true under both losses so far tried: naively adding more raw-categorical fields to this
FM does not help, and mildly hurts once the vocab grows past ~9 fields. The likely mechanism is that
each added field is one more high-cardinality (or low-signal, e.g. `_range`-bucketed) embedding table
diluting the same fixed interaction capacity (k=16), not adding usable signal — this is a capacity/
regularization problem, not a "more data always helps" one. Do not spend an iteration re-deriving this
by adding one CWM field at a time; if a feature idea from this specific field list resurfaces, it
needs a reason to work differently now (e.g. a numeric/log-scaled encoding instead of a `_range`
bucket, a wider `k`, or an interaction-aware architecture with its own regularization) — not just
"try it and see," which has now been tried and answered twice.

### [curated] KuaiRand dataset (background, not a method)  (C. Gao et al., "KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos," CIKM 2022 (arXiv 2208.08696))
**What it is**: a Kuaishou short-video interaction log built specifically to include a slice of *randomly*-exposed impressions (not chosen by any recommender), enabling off-policy/counterfactual evaluation research that a purely algorithm-selected log can't support. The normal `log_standard` files (what this project's `workspace/data/` actually contains) are algorithm-selected and carry the usual exposure/selection bias of any production log; a **separate** `log_random_*.csv` file holds the randomly-exposed rows, and was **not** downloaded into this project.

**Why this matters directly**: this project's own EDA found `is_rand == 0.0` on every row across all splits — this is exactly why: the randomized-exposure log simply isn't part of the local dataset, not a data quality problem. Any method that specifically needs the randomized log (e.g. IPS-based debiasing, see `popularity_debias.md`) is not usable without downloading `log_random_*.csv` separately.

**Confirmed field meanings** (from the dataset's own documentation, cross-checked against this project's EDA numbers):
- `tab`: which of 15 values (0-14) UI scenario/feed placement the impression came from (e.g. main feed vs. other app pages) — matches this project's EDA finding of exactly 15 distinct `tab` values in train; it's a real categorical signal, not an encoding artifact.
- `video_type`: takes values `NORMAL` or `AD`.
- `tag`: documented upstream as "a list of key categories" for a video; this project's `video_features_basic_pure.csv` copy stores a single numeric id per row — treat any multi-tag semantics as unconfirmed for this specific file.

**Scale**: KuaiRand-Pure = 27,285 users x 7,583 videos — matches this project's EDA cardinality numbers exactly. The organizer starter kit's 1.4M-row train+eval set (dates 4/08-5/08) is a fixed date-windowed subset of the full ~2.6M-row Pure `log_standard`, not the complete released collection — an intentional starter-kit choice, not missing data.