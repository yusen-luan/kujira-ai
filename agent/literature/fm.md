---
title: Factorization Machines
citation: S. Rendle, "Factorization Machines," ICDM 2010
tags: factorization machine FM pairwise interaction embedding baseline CTR ranking sparse categorical low-rank
---

**Core idea**: Model every pairwise (2nd-order) feature interaction with a dot product of two low-rank latent vectors, one per feature value, instead of a full pairwise weight matrix. This makes the interaction term cost O(k·n) instead of O(n^2) for n features and embedding size k, and — critically for sparse high-cardinality ID features like `user_id`/`video_id` — lets the model estimate an interaction between two feature values that never co-occurred together in training, by transitivity through shared latent factors. A plain logistic regression with hand-crossed features can't do this at all for unseen combinations.

**When it helps**: essentially always a reasonable starting point for sparse categorical CTR-style data — it's the reason this project's official baseline is FM, not a deeper model. Also the right thing to try first whenever a deeper model underperforms it: if DeepFM/DCN/etc. can't beat plain FM, the extra capacity likely isn't buying anything on this feature set yet, and the fix is more/better features, not a bigger model.

**Cost**: cheapest option by far — the whole starter kit's FM baseline runs in ~40s on one CPU core.

**KuaiRand-Pure notes**: this is the fixed official baseline this project is scored against (validation primary 0.6016, hidden test 0.5946). Every other candidate here is implicitly being compared to it.
