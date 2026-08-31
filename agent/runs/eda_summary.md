## Label & class balance
- Pipeline target `long_view` is moderately imbalanced (~33.7% train, ~31.3% eval) — no resampling needed but watch calibration.
- Other engagement labels (`is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`) are extremely rare (<3%) — not viable as standalone targets without heavy imbalance handling; `is_rand` is all-zero (unusable).
- `is_click` (~45%) could serve as an auxiliary/pretraining task given its higher base rate and correlation with engagement funnel.

## Popularity & cold-start
- Severe video popularity skew: top 1% of videos account for ~24% of positives (Gini 0.79) — long-tail videos need frequency/popularity features or exposure-debiasing.
- User engagement is highly heterogeneous: ~30% of eval users have zero positives, ~12% are all-positive — per-user history features (e.g., historical long_view rate) are strongly predictive (quintile spread 0.14→0.55) and should be added.
- Cold-start is user-side, not item-side: unseen users ~1.6% (valid)/3.6% (test); unseen videos near-zero (<0.02%) — prioritize user cold-start handling (e.g., fallback to `user_features_pure.csv` demographics), item cold-start is not a current concern.

## Data quality issues
- `user_features_pure.csv.is_live_streamer` has 21,127 rows with negative values (e.g., -124) in an apparent flag/count column — treat as missing/sentinel, not literal value; needs cleaning/imputation before use.

## Leakage risk
Never use these same-row raw log columns as input features — they are post-exposure outcomes/durations only known after the impression is served and directly encode or are downstream of the label:
- `play_time_ms`, `profile_stay_time`, `comment_stay_time` — direct post-click engagement duration signals used to derive `long_view` and other labels.
- `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `is_profile_enter` — other same-session engagement outcomes, target-leaking for `long_view`.

## Distribution shift
- `tab` distribution shifts materially from train→test (tab=0 share drops 13%→8%, tab=6 rises 2.6%→4.7%) — treat `tab` as a shift-sensitive feature; consider train/test-invariant encoding or reweighting.
- Mean video `duration_ms` drifts upward across splits (train 97.9k → test 107.2k ms) — monitor duration-based features for drift, though duration itself is only weakly correlated with `long_view` (see below).
- `long_view` base rate itself drifts slightly down (33.7%→31.3%) — expect minor calibration shift at eval time.

## Unused data available
- `user_features_pure.csv`: `user_active_degree`, `follow/fans/friend_user_num(_range)`, `register_days(_range)`, `onehot_feat0-17` — not read by pipeline; `user_active_degree` alone shows large rate spread (0.14–0.44), strong candidate categorical feature.
- `video_features_basic_pure.csv`: `author_id`, `video_type`, `upload_type`, `tag`, `video_duration`, `music_id/type`, `server_width/height` — not read; `tag`/`upload_type` show large rate spreads (tag: 0.07–0.47, upload_type: 0.06–0.37) and should be added as categorical features; `author_id` enables author-level popularity/quality aggregates.

## Agent's own follow-up investigations
- Confirmed `user_active_degree`, `tag`, `upload_type` (and to lesser extent `video_type`) have large, non-noise long_view rate spreads — high-value categorical features to add from currently-unused side files.
- User historical long_view rate (leave-out-valid) strongly separates outcomes (quintile rates 0.14→0.55) — add as a leakage-safe (prior-split-only) feature.
- Session position within a user's train sequence shows a real monotonic decline in long_view rate (0.41 at position 0 → 0.31 at 21+) — likely fatigue effect; consider adding impression-position/fatigue feature.
- Video duration shows only a weak, non-monotonic relationship with long_view (rate spread ~0.10, Pearson ≈0.003) — deprioritize raw duration as a strong standalone feature despite intuition; the play_time/duration *ratio* (leakage-safe only post-hoc, not for input) is far more separating and confirms label construction logic, not a feature to add.