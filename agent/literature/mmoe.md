---
title: MMoE (Multi-gate Mixture-of-Experts)
citation: J. Ma, Z. Zhao, X. Yi, et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts," KDD 2018
tags: MMoE multi-gate mixture of experts multi-task learning task relationship seesaw gating shared bottom
---

**Core idea**: An alternative multi-task architecture to ESMM's sequential-multiplicative structure. A bank of shared "expert" sub-networks all see the same input; each task has its own small gating network that learns a soft weighting over which experts to draw from for *that* task. Unlike a plain shared-bottom model (one shared trunk, task-specific heads only bolted on top), each task effectively gets its own mixture of the shared experts — so tasks that don't transfer well to each other can down-weight the experts that aren't useful to them. This directly targets the "seesaw problem": jointly training dissimilar tasks with a single shared trunk can make one task's gradient actively hurt another's.

**When it helps**: when jointly training several of the 12 feedback signals at once (not just a strict click-then-conversion chain like ESMM assumes) and the tasks are suspected to not all transfer equally well — the gating lets the model discover per-task relatedness rather than assuming one shared representation serves every signal equally well.

**Cost**: moderate — scales with (number of experts) x (per-expert network size) plus small per-task gating networks; tune both down to fit the CPU wall-clock budget.

**KuaiRand-Pure notes**: `AGENT.md` notes the model zoo already includes `Share_*` shared-bottom multi-task variants (a plain hard-shared trunk, no gating) from a prior project. MMoE is the natural next step up from those specifically if a shared-bottom multi-task attempt shows the seesaw problem in practice (one task's validation gain coming at another's expense).
