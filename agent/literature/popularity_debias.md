---
title: Popularity bias & debiasing
citation: survey synthesis: M. Klimashevskaia et al., "A Survey on Popularity Bias in Recommender Systems," arXiv 2308.01118 (2023); J. Chen et al., "Bias and Debias in Recommender System: A Survey and Future Directions," arXiv 2010.03240 (2020)
tags: popularity bias debias exposure bias long tail gini re-weighting inverse propensity scoring IPS negative sampling
---

**Core idea**: Logged interaction data over-represents popular items because *exposure itself* is already biased by whatever recommender served the historical logs — popular items get shown, and hence get positive labels, far more often regardless of true per-user relevance. A model trained naively on this data partly relearns "what was popular" rather than "what's relevant." Standard mitigation families: (a) re-weight training examples inversely to item/exposure popularity, (b) regularize against or explicitly subtract a learned popularity-only signal from the main score, (c) causal / inverse-propensity-scoring (IPS) methods that reweight by an estimated exposure probability (need a source of unbiased/randomized exposure to estimate propensities from), (d) popularity-aware negative sampling.

**When it helps**: when a model's ranking looks like it's mostly rediscovering item popularity rather than adding personalization signal beyond it — this project's own EDA found exactly this pattern (Gini=0.79 on video-level positive counts, top 1% of videos = ~24% of all positives), and the `pop`-only baseline already scores uncomfortably close to the FM baseline (0.5715 vs 0.6016 primary) — evidence popularity alone already explains a large share of the achievable score here.

**Cost**: (a)/(d) are close to free — a training-loop reweighting or sampling change, no architecture change needed; (b)/(c) need an auxiliary popularity or propensity estimate but are still lightweight relative to a full model swap.

**KuaiRand-Pure notes**: IPS-style methods (c) specifically need the randomized-exposure log to estimate unbiased propensities — see `kuairand_dataset.md`: this project's downloaded data has `is_rand == 0.0` on every row, because the separate `log_random_*.csv` file wasn't part of what got downloaded. (a)/(b)/(d) have no such dependency and remain usable as-is.
