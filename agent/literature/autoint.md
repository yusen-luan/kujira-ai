---
title: AutoInt
citation: W. Song et al., "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks," CIKM 2019
tags: AutoInt self-attention multi-head feature interaction embedding CTR interpretable attention weights
---

**Core idea**: Map every field into a shared embedding space, then apply multi-head self-attention across the field embeddings (with residual connections) so the model learns *which* feature pairs interact and *how strongly*, per example, rather than FM's fixed uniform pairwise term or DCN's fixed cross-layer structure. Stacking attention layers lets higher-order combinations emerge without hand-designed crossing. Attention weights give a rough interpretability signal (which fields the model attended to most for a given prediction).

**When it helps**: worth trying when the useful interactions are likely input-dependent rather than uniform across all examples — e.g. which pair of fields matters might itself depend on context (tab, duration bucket). Best tried once a fixed-structure model (FM/DeepFM/DCN) has plateaued, since attention needs enough fields to have something to select among.

**Cost**: heavier than FM/DeepFM/DCN in principle (attention cost grows with num_fields²), but with only ~5-13 fields in this project that's still a small number of pairs (<200) — modest `atten_embed_dim`/`num_heads`/`num_layers` should stay within the CPU wall-clock budget already used for the model axis.

**KuaiRand-Pure notes**: `workspace/models/afi.py` already adapts this (named for "Automatic Feature Interaction"). Most useful once the feature axis has widened the field count beyond the starter kit's 5 — more fields give attention more to actually select among.
