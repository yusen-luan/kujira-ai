---
title: Naive field-widening does not improve FM on this data (ablation, both loss functions)
citation: this project's own ablation, workspace/ablation_features.py (verified 2026-08-31 under both the original pointwise-BCE loss and the current best's BPR pairwise loss)
tags: feature interaction, CTR prediction, ranking, embedding, feature addition, ablation, field count, vocab size, overfitting, capacity, FM
source: project
---
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
