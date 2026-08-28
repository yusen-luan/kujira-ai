---
title: ESMM (Entire Space Multi-Task Model)
citation: X. Ma et al., "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate," SIGIR 2018
tags: ESMM entire space multi-task CVR CTR sample selection bias sparse sparsity conversion sequential dependency auxiliary task
---

**Core idea**: Built for a chain of dependent binary outcomes (impression -> click -> conversion). Two towers share one embedding table: a CTR-style tower predicts P(click | impression) over *every* impression; a CVR-style tower's raw output is never trained or scored on its own — it's only ever multiplied by the CTR tower's probability to directly estimate P(click AND next-event) over the whole impression space. This sidesteps two problems a naive "train a classifier only on clicked rows" approach hits: sample-selection bias (the downstream model would only ever see clicked examples during training, but must score every impression at inference) and extreme sparsity of the downstream positive label.

**When it helps**: exactly the pattern this project's own EDA flagged — `is_like`/`is_follow`/`is_comment`/`is_forward` are all under 3% positive and are logically downstream of engagement (unlikely to comment on something never really watched). Training a standalone classifier directly for one of these hits the same sample-selection-bias-plus-sparsity problem ESMM was designed for. `long_view` (this project's actual label) could play the role of the upstream "click" stage; a rarer signal like `is_like` or `is_comment` could play the downstream "conversion" stage.

**Cost**: two small MLP towers sharing one embedding table — roughly 2x a single-task model's cost, no separate downstream-labeled dataset needed since both stages are logged on the same rows here.

**KuaiRand-Pure notes**: no existing implementation in `workspace/models/*.py` — the zoo's variants are single-task or shared-bottom multi-task, not this specific sequential-multiplicative structure, so this would need new code rather than adapting an existing file.
