---
title: Deep & Cross Network (DCN, DCN-V2)
citation: R. Wang et al., "Deep & Cross Network for Ad Click Predictions," ADKDD 2017; R. Wang et al., "DCN V2," WWW 2021
tags: DCN deep cross network explicit feature interaction bounded degree cross layer low-rank web-scale
---

**Core idea**: Replace (or complement) an implicit MLP with an explicit "cross layer" stacked L times: each layer computes `x0 ⊙ (W·x_l) + b_l + x_l`, so the polynomial degree of feature interactions the network can represent grows linearly with depth L, explicitly and with far fewer parameters than an MLP would need to approximate the same interactions implicitly. The original 2017 DCN uses a per-layer weight *vector*; DCN-V2 (2021) generalizes this to a full (or low-rank-factorized, for efficiency) weight *matrix* per layer, and reports this materially improves accuracy at web scale versus the vector form.

**When it helps**: when you suspect useful interactions go beyond pairwise but are still fairly low/bounded degree, and you'd rather have that structure explicit than hope an MLP discovers it implicitly. Note the ceiling on how much this can help is set by how many fields you actually have — with only the starter kit's 5 fields, the interaction space is already small, so cross-layer depth has limited room to add value until the feature set is widened (pair with a feature-axis change first).

**Cost**: cheap — cross layers are lightweight compared to a wide MLP or attention.

**KuaiRand-Pure notes**: `workspace/models/dcn.py` adapts torchfm's `CrossNetwork`, which implements the original 2017 vector-weight form, not DCN-V2's matrix form — upgrading to DCN-V2-style per-layer matrices (or a low-rank factorization of them) is a legitimate, currently-unexplored model-axis hypothesis if the vector form underfits.
