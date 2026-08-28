---
title: DeepFM
citation: H. Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction," IJCAI 2017
tags: DeepFM wide deep neural network CTR ranking embedding shared factorization machine implicit interaction
---

**Core idea**: Run FM's 2nd-order term and a plain DNN in parallel over the *same* shared field embeddings, then sum their outputs. The FM side keeps the well-understood, easy-to-fit low-order signal; the DNN side learns higher-order interactions implicitly, without the manual "wide-side" feature crossing that Wide&Deep (its predecessor) needed. Because both towers read the same embedding table, this is cheap to add — it's not really a new model so much as FM plus a free extra head.

**When it helps**: a natural first thing to try once a plain FM plateaus, if you suspect there's signal in interactions higher than pairwise (e.g. a 3-way combination of tab × duration-bucket × author) that FM's pairwise-only structure can't represent. Because it keeps the FM term, it should very rarely do *worse* than FM alone in principle — if it does worse in practice on a small dataset, that's usually the DNN side overfitting, not a fundamental issue with the architecture.

**Cost**: small — one MLP added on top of an embedding table you already need for FM; cheap enough to be a same-order-of-magnitude increase in training time over plain FM.

**KuaiRand-Pure notes**: `workspace/models/dfm.py` already adapts this from a prior project, reusing the same field-offset embedding convention `data.py`'s `encode()` produces — the most direct "try a deeper model" experiment available on the model axis.
