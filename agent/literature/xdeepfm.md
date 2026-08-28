---
title: xDeepFM
citation: J. Lian et al., "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems," KDD 2018
tags: xDeepFM compressed interaction network CIN vector-wise explicit implicit feature interaction
---

**Core idea**: Adds a Compressed Interaction Network (CIN) alongside a plain DNN. CIN builds explicit interactions at the *vector* level — whole embedding vectors combine as units, like a structured, compressed outer-product across layers — whereas a plain DNN mixes features implicitly at the *bit* level (individual embedding dimensions get tangled together with no per-field structure preserved). The parallel DNN still captures arbitrary implicit interactions the same way DeepFM's DNN side does; CIN's explicit vector-wise structure is the actual addition. Positioned as a generalization that subsumes DeepFM's wide-and-deep split while adding a genuinely new explicit-interaction mechanism (distinct from DCN's cross layers) rather than duplicating it.

**When it helps**: when you suspect explicit, higher-than-pairwise interactions matter (3rd order or beyond) and DCN's cross layers — also explicit, but scalar/matrix-weighted rather than structured per-field like CIN — haven't captured them. A reasonable next step after DeepFM and DCN have both been tried.

**Cost**: the heaviest of this family — CIN's compute/memory scales with `cross_layer_sizes` (the feature-map width per layer) × number of fields, so keep `cross_layer_sizes` modest to fit the model axis's CPU/wall-clock budget; watch `--model_run_timeout` specifically.

**KuaiRand-Pure notes**: `workspace/models/xdfm.py` already adapts torchfm's `CompressedInteractionNetwork`. Try after DeepFM/DCN if those plateau — it's the priciest option on the model axis, so budget for it accordingly.
