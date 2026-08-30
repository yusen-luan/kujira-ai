## Label & class balance
- Target `long_view` is fairly balanced (~31-34% positive) across splits — no resampling/class-weighting needed.
- Other engagement labels (`is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`) are extremely rare (<3%) — not viable as standalone targets without heavy imbalance handling; could serve as auxiliary/multi-task signals instead.

## Popularity & cold-start
- Strong video popularity skew: top 1% of videos drive ~24% of positives, Gini=0.79 — item popularity/frequency features likely high-value; watch for popularity bias in ranking.
- Video cold-start is negligible (<0.02% unseen in valid/test) — no need for content-based cold-start handling on items.
- User cold-start is non-trivial and grows over time (1.6% unseen in valid → 3.6% in test) — user-side model should degrade gracefully for new users (e.g. fallback to content/popularity features).
- ~30% (valid) / 27% (test) of users have zero positives and only ~58-64% are GAUC-eligible — per-user eval metrics will be noisy/biased for low-activity users; consider filtering or weighting when reporting per-user metrics.

## Data quality issues
- `user_features_pure.csv.is_live_streamer` has 21,127 rows with negative values (e.g. -124) — these are sentinel/missing codes, not real counts; must be cleaned (mask or impute) before use as a feature.

## Leakage risk
Never use these same-row raw log columns as input features — they are outcomes/artifacts of the very interaction being predicted (post-hoc signals, only known after impression is served/watched):
- `play_time_ms`, `profile_stay_time`, `comment_stay_time` — direct measures of user engagement duration during the event.
- `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `is_profile_enter` — co-occurring engagement outcomes on the same row as the label.

## Distribution shift
- `tab=0` share shrinks train→valid→test (13%→11%→8%) while `tab=1` grows (73%→74%→77%) — tab/surface distribution drifts over time; treat `tab` as time-sensitive and consider recency-aware validation.
- Mean video `duration_ms` rises slightly across splits (98k→103k→107k) — mild content-length drift, unlikely to require action alone but corroborates temporal drift.
- `long_view` rate drifts down slightly train→eval (33.7%→31.3%) — consistent with temporal shift; monitor calibration on later time periods.

## Unused data available
- `user_features_pure.csv`: `user_active_degree`, `follow/fans/friend_user_num(_range)`, `register_days(_range)`, `is_live_streamer`, `is_video_author`, `onehot_feat0-17` — rich user profile/social-graph features not joined in; could add as static user embeddings/features to address cold-start and personalization.
- `video_features_basic_pure.csv`: `author_id`, `video_type`, `upload_dt`, `upload_type`, `visible_status`, `video_duration`, `server_width/height`, `music_id`, `music_type`, `tag` — item metadata not joined in; `author_id` enables author-level popularity/CF features, `tag`/`video_type` enable content-based similarity for cold-start items, `upload_dt` enables video-age/freshness features.