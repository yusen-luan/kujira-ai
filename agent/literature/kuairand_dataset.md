---
title: KuaiRand dataset (background, not a method)
citation: C. Gao et al., "KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos," CIKM 2022 (arXiv 2208.08696)
tags: KuaiRand dataset tab is_rand log_random randomized exposure counterfactual off-policy video_type tag cardinality Kuaishou
---

**What it is**: a Kuaishou short-video interaction log built specifically to include a slice of *randomly*-exposed impressions (not chosen by any recommender), enabling off-policy/counterfactual evaluation research that a purely algorithm-selected log can't support. The normal `log_standard` files (what this project's `workspace/data/` actually contains) are algorithm-selected and carry the usual exposure/selection bias of any production log; a **separate** `log_random_*.csv` file holds the randomly-exposed rows, and was **not** downloaded into this project.

**Why this matters directly**: this project's own EDA found `is_rand == 0.0` on every row across all splits — this is exactly why: the randomized-exposure log simply isn't part of the local dataset, not a data quality problem. Any method that specifically needs the randomized log (e.g. IPS-based debiasing, see `popularity_debias.md`) is not usable without downloading `log_random_*.csv` separately.

**Confirmed field meanings** (from the dataset's own documentation, cross-checked against this project's EDA numbers):
- `tab`: which of 15 values (0-14) UI scenario/feed placement the impression came from (e.g. main feed vs. other app pages) — matches this project's EDA finding of exactly 15 distinct `tab` values in train; it's a real categorical signal, not an encoding artifact.
- `video_type`: takes values `NORMAL` or `AD`.
- `tag`: documented upstream as "a list of key categories" for a video; this project's `video_features_basic_pure.csv` copy stores a single numeric id per row — treat any multi-tag semantics as unconfirmed for this specific file.

**Scale**: KuaiRand-Pure = 27,285 users x 7,583 videos — matches this project's EDA cardinality numbers exactly. The organizer starter kit's 1.4M-row train+eval set (dates 4/08-5/08) is a fixed date-windowed subset of the full ~2.6M-row Pure `log_standard`, not the complete released collection — an intentional starter-kit choice, not missing data.
