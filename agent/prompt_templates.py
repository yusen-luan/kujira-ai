"""Prompts + response parsing for the propose/repair LLM calls.

v2 (2026-08-29, replaces the v1 feature/model axis-tree): a single unified
system prompt and a single linear iteration chain. Each turn, the LLM sees the
*whole* current pipeline (data.py's feature engineering AND baseline.py's
model/training procedure together, not a pre-assigned axis) and picks
whichever single change it judges highest-leverage this turn -- a feature
change, a model/architecture change, or a loss/training-procedure change, all
governed by the same "one attributable change per turn" rule that v1's axes
enforced structurally. This fixed a real bug in v1: the model axis was seeded
independently from workspace/ and never picked up the feature axis's accepted
improvements, so every model-architecture hypothesis was tested against a
strictly worse feature set than what was actually available. See
agent_notes/orchestrator.md for the full design discussion.

Four possible actions per turn (mutually exclusive, decided by the LLM itself,
not an external classifier call):
  1. Propose a hypothesis + code change (REFLECTION + HYPOTHESIS +
     EXPECTED_EFFECT + a data.py and/or baseline.py fenced block).
  2. Write a small read-only diagnostic probe instead, if it doesn't have
     enough information to propose a good hypothesis this turn (REFLECTION +
     PROBE_QUESTION + a probe.py fenced block). Replaces v1's separate
     diagnosis-classification call + separate probe-writing call with one
     call that can just do either -- cheaper and simpler, and the LLM is
     already the one deciding whether it's stuck, not a second LLM call
     re-deciding the same thing from a summary.
  3. Ask a live web-research question instead (REFLECTION + RESEARCH_QUESTION,
     no code), if it suspects a specific published method would help but the
     curated literature/EDA/probe context below doesn't already cover it.
     Routed to a separate, tool-restricted LLM call (agent/web_research.py,
     via llm.call_claude_research()) that can only WebSearch/WebFetch --
     never execute code -- and whose output is a citation-checked note merged
     into the same local corpus agent/rag.py retrieves from, never applied as
     code directly. Drawn from a small separate --web_research_budget pool
     (like the old --diagnosis_budget), not from --iterations; see the budget
     line in the propose prompt for how many are left this run.
  3b. Ask an EXPLORE_QUESTION instead (REFLECTION + EXPLORE_QUESTION, no
     code) -- the same idea as RESEARCH_QUESTION, but pointed at THIS repo
     instead of the web: "does something in this codebase already answer
     this?" (an existing script, a prior write-up, unused reference code).
     Added after a real gap: workspace/ablation_features.py already tested
     whether widening the encoded field set helps, and the propose role had
     no way to discover that on its own -- it could only ever see what
     agent/eda.py's one-time pass and the curated literature happened to
     paste into its prompt. Routed to agent/repo_explore.py (via
     llm.call_claude_explore()), which grants ONLY Read/Grep/Glob, confined
     to this repo's root -- never Bash/Edit/Write, so it can look but never
     act. Its output is a citation-checked note (every claimed file path is
     verified to actually exist under the repo root) merged into the same
     local corpus agent/rag.py retrieves from, same as a research note.
     Drawn from a small separate --explore_budget pool.
  3c. Ask for a new EDA_ROUND_REQUEST instead (REFLECTION + EDA_ROUND_REQUEST,
     no code) -- v4 roadmap Phase 2. agent/eda.py's deterministic report and
     first agentic-EDA pass run once at startup, but the propose LLM itself
     can request ANOTHER round mid-run if it suspects recent lack of progress
     is a *data-understanding* gap rather than a *modeling* gap -- the streak/
     escalation context already in this prompt is exactly the signal it needs
     to make that call. Routed to agent/eda_agent.py's exploration loop (write
     a probe.py -> run it through eda_probe.py's fixed harness, same
     mechanism PROBE_QUESTION above uses, via agent/probe_runner.py -- but
     genuinely iterative: it can look at one result and decide to look
     further, not just repair-on-failure), whose accumulated findings get
     folded into a freshly regenerated eda_summary.md so they persist into
     every future propose prompt without needing to be re-discovered. Drawn
     from a small separate --eda_round_budget pool.
  3d. Ask for a CODE_SESSION_REQUEST instead of committing to a one-shot full-file
     hypothesis -- v4 roadmap Phase 4. Every option above still hands the LLM zero
     tools -- it proposes a full replacement file body once, blind, and only finds
     out if it runs via the repair loop after the fact. This option instead routes
     to agent/code_agent.py (via llm.call_claude_code()), a bounded MULTI-TURN
     session that grants Read/Grep/Glob/Edit/Write -- genuinely confined to one
     candidate node directory under --restricted (verified live: a cwd-confined
     session was refused on a relative-path read one directory outside it) -- so it
     can read the current data.py/baseline.py, edit them directly, and iterate
     before handing off, rather than guessing a whole file at once. Deliberately no
     Bash grant: also verified live that --restricted's cwd-confinement does NOT
     extend to Bash (a restricted session with Bash granted read a file outside its
     cwd via `cat ../...` without issue) -- Bash's actual reach is the OS user's
     full filesystem regardless of cwd, fine for a human-watched interactive
     session but not for this orchestrator's unattended runs, so this option trades
     away in-session diagnostics (already covered by EXPLORE_QUESTION/PROBE_QUESTION
     above) for keeping the same real confinement guarantee every other tool-granted
     role here relies on. The session ends with the same REFLECTION/HYPOTHESIS/
     EXPECTED_EFFECT/LEVER_CATEGORY text contract as option 1, just with no fenced
     file blocks -- the files are already edited on disk, so orchestrator.py reads
     them straight from the candidate directory afterward and feeds the result
     through the exact same run_and_repair() -> accept/reject path option 1 uses,
     rather than a parallel status family. Drawn from a small separate
     --code_session_budget pool. Not the default/common path -- worth the extra
     cost mainly for a change gnarly enough that reading-while-editing genuinely
     helps, not a small, easily-specified-in-one-shot change.
  4. Race MAX_VARIANTS (default 3) structurally different hypothesis+code
     variants in parallel instead of committing to one (REFLECTION + a
     `VARIANT N:`-tagged block per idea, each with its own HYPOTHESIS/
     EXPECTED_EFFECT/files). orchestrator.py runs every variant concurrently
     (each its own subprocess, no repair-retry -- a bad variant just fails,
     cheaper than repairing N of them), keeps whichever beats current best by
     the most, and folds EVERY variant's hypothesis + outcome -- including the
     losing ones -- into history, so a losing variant's insight still reaches
     the next iteration instead of being thrown away. Still exactly ONE
     accept/reject decision and one entry in the convergence trace per node,
     no matter how many variants raced to produce it -- parallelism here is
     about what's inside a node, not about redefining what a node is. Free on
     the LLM-token axis (still one propose call) since this benchmark trains
     on CPU in well under a couple minutes; the cost is pure wall-clock/CPU,
     which is cheap here specifically.

Plateau escalation (replaces v1's --stuck_after/diagnosis_budget machinery):
orchestrator.py tracks a plain consecutive-iterations-without-a-*significant*-
improvement streak (using the official epsilon=--converge_eps). Accept/reject
is a separate decision from significance -- any node that beats the current
best at all is accepted into best_dir, regardless of margin, so best_dir
always holds the actual best-ever score; only whether the margin cleared
epsilon feeds this streak. Once it crosses --escalate_after, the propose
prompt gets an explicit "you're plateauing, don't propose a variant of what
you just tried" instruction, naming the recent hypotheses so the LLM has
something concrete to diverge from. Probes consume a normal --iterations
slot now (no separate reserved budget) -- simpler, and self-limiting since
spending the whole run on probes instead of code changes doesn't move the
metric either.

Lever-category tracking (added after a real run repeatedly cycled through
"add a feature" / "tune a training knob" variants without ever trying
popularity-debiasing, multi-task, or data-quality/preprocessing changes,
despite the EDA and literature corpus mentioning all three every turn):
every hypothesis and every VARIANT now carries a required LEVER_CATEGORY tag
from a fixed taxonomy (see LEVER_CATEGORIES below). orchestrator.py's
history already records every attempt; `_tried_categories()` scans it and
`_format_escalation()` explicitly lists which categories have NEVER been
tried yet once plateaued, instead of a generic "try something different" --
the point isn't just diversity for its own sake, it's steering away from
categories that have already failed toward ones that haven't been tested at
all, since blind diversity can still re-explore an already-exhausted
category by accident (as happened for four straight "add a feature" /
"tune a knob" turns in the run that motivated this).
"""
import re
from pathlib import Path

ALLOWED_FILES = ('data.py', 'baseline.py')
PROBE_ALLOWED_FILES = ('probe.py',)
MAX_SWEEP_VALUES = 4
MAX_VARIANTS = 3

# Fixed taxonomy every hypothesis/variant must self-tag with (LEVER_CATEGORY:), so
# orchestrator.py can track which categories have actually been tried -- not just
# which specific hypotheses -- and steer the plateau nudge toward untried categories
# instead of a vague "try something different" that in practice kept re-picking
# feature_addition/training_procedure/hyperparameter_capacity over and over. Kept
# short and mutually exclusive by construction; 'other' is a legitimate explicit
# choice for a genuinely novel idea that doesn't fit -- 'uncategorized' (not in this
# list) is reserved for a missing/unparseable tag, never something the LLM should
# choose itself.
LEVER_CATEGORIES = (
    'feature_addition',
    'data_quality_preprocessing',
    'model_architecture',
    'training_procedure',
    'negative_sampling',
    'popularity_debiasing',
    'multi_task',
    'hyperparameter_capacity',
    'ensembling',
    'other',
)
_LEVER_CATEGORY_BLURBS = {
    'feature_addition': 'adding/removing/rebucketing raw or engineered columns in data.py',
    'data_quality_preprocessing': 'cleaning sentinel/missing values, changing the encoding scheme, '
        'cold-start-robust fallbacks, temporal-drift-aware features or sample weighting, deduplication',
    'model_architecture': "changing the scoring function/model family (e.g. FM -> DeepFM/AutoInt/xDeepFM)",
    'training_procedure': 'loss function, optimizer, epochs/patience/regularization strength',
    'negative_sampling': 'how negatives are chosen for a pairwise/BPR-style training objective',
    'popularity_debiasing': 'correcting for item-popularity skew in features, loss, or scoring '
        '(see the EDA popularity-skew finding and the popularity-debiasing literature note)',
    'multi_task': 'jointly modeling multiple feedback signals (e.g. ESMM/MMoE-style) instead of '
        'training on long_view alone',
    'hyperparameter_capacity': 'sweeping an EXISTING knob (k, dropout, mlp width, ...) without '
        'changing the underlying mechanism -- use this only when SWEEP_PARAM is the whole idea',
    'ensembling': 'combining predictions from 2+ mechanisms you (or a past iteration) already tried '
        '-- e.g. average two models\' scores -- in ONE run_model() call, rather than proposing a '
        'brand-new mechanism; see the ranked past-candidates list below for what\'s worth combining. '
        'Pick candidates that differ in MECHANISM (different lever_category, e.g. a model_architecture '
        'change paired with a training_procedure change), not just the two highest-scoring ones -- node '
        '17 ensembled plain BPR-FM with the same FM\'s auxiliary-head variant (same base mechanism, '
        'differing only by one small added loss term) and got essentially the same score either way '
        '(0.6039 alone or ensembled): near-identical models make near-identical errors, so averaging '
        'them barely reduces variance. A lower-scoring but structurally different candidate is usually a '
        'better ensembling partner than the next-highest score.',
    'other': "a genuinely novel idea that doesn't fit any category above",
}


def _normalize_lever_category(raw):
    key = (raw or '').strip().lower().replace(' ', '_').replace('-', '_')
    return key if key in LEVER_CATEGORIES else 'uncategorized'


_LEVER_TAXONOMY = ("Lever categories — every hypothesis (or VARIANT) must self-tag with exactly one "
                    "of these via LEVER_CATEGORY (see output format below); pick whichever actually "
                    "describes the change, not whichever sounds most novel:\n" +
                    "\n".join(f"- {cat}: {_LEVER_CATEGORY_BLURBS[cat]}" for cat in LEVER_CATEGORIES))

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_ZOO_DIR = _REPO_ROOT / 'workspace' / 'models'
_MODEL_ZOO_FILES = ('fm.py', 'dcn.py', 'dfm.py', 'afi.py', 'xdfm.py')
_EDA_SUMMARY_PATH = _REPO_ROOT / 'agent' / 'runs' / 'eda_summary.md'
_LITERATURE_CONTEXT_PATH = _REPO_ROOT / 'agent' / 'runs' / 'literature_context.md'
_PROBE_FINDINGS_PATH = _REPO_ROOT / 'agent' / 'runs' / 'probe_findings.md'


def _load_literature_context():
    if _LITERATURE_CONTEXT_PATH.exists():
        text = _LITERATURE_CONTEXT_PATH.read_text(encoding='utf-8').strip()
        if text:
            return text
    return ('(no literature retrieved -- run `python agent/rag.py` first, or it runs '
            'automatically before the first propose call in orchestrator.py.)')


def _load_eda_summary():
    if _EDA_SUMMARY_PATH.exists():
        text = _EDA_SUMMARY_PATH.read_text(encoding='utf-8').strip()
        if text:
            return text
    return ('(no EDA report found -- run `python agent/eda.py` first. Proceeding '
            'without data-grounded facts; treat any assumption about class balance, '
            'cardinality, or missing-value handling as unverified.)')


def _load_probe_findings():
    if _PROBE_FINDINGS_PATH.exists():
        text = _PROBE_FINDINGS_PATH.read_text(encoding='utf-8').strip()
        if text:
            return text
    return '(no diagnostic probes have been run yet this project.)'


def _load_model_zoo_reference():
    """Inlines the torch model zoo source. llm.py calls the LLM with --tools ""
    (pure text completion, no filesystem access), so anything it should be able
    to adapt has to be pasted into the prompt text."""
    parts = []
    for fname in _MODEL_ZOO_FILES:
        path = _MODEL_ZOO_DIR / fname
        if not path.exists():
            continue
        parts.append(f"--- workspace/models/{fname} ---\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


_TORCH_AVAILABLE_LINE = ("- numpy and the Python standard library only in data.py. torch and torchfm are "
    "additionally allowed in baseline.py (not in data.py) if you're proposing a model/architecture "
    "change — both are installed. No other new pip dependencies, no internet access.")
_TORCH_UNAVAILABLE_LINE = ("- numpy and the Python standard library only, in both data.py and "
    "baseline.py. torch and torchfm are NOT importable in this run's environment (checked at "
    "startup) — do NOT propose a hypothesis that imports either; a model/architecture change must "
    "be implemented in plain numpy instead. No other new pip dependencies, no internet access.")


def _build_contract(torch_available):
    torch_line = _TORCH_AVAILABLE_LINE if torch_available else _TORCH_UNAVAILABLE_LINE
    return f"""Hard constraints you must preserve, because a fixed harness calls into these files \
directly and cannot be changed:
- data.py must keep a `load(data_dir)` function returning the same split dict shape, \
and an `encode(splits)` function returning `(enc, dim)` where `enc[name] = (X, y, users)`.
- baseline.py must keep a `run_model(splits, hparams: dict, seed=0, verbose=True)` \
function returning `{{'valid': {{...}}, 'test': {{...}}}}` (each a dict with GAUC, nDCG@5, \
primary) — this is the fixed entrypoint the harness calls, so it must exist with this \
exact name and signature no matter what else you change. The harness always passes the \
same generic hparams dict regardless of what kind of change you're making, so don't assume \
it contains any particular keys — read what you need via `hparams.get(name, default)` with \
sensible defaults, and don't raise if a key you'd expect is missing or one you don't use \
is present.
{torch_line}
- Mind the wall-clock budget: any candidate that touches baseline.py's training procedure must \
finish within the run timeout on CPU (no GPU available) — keep epoch count and model size modest. \
A smaller/faster model that actually finishes and beats the current best is better than a bigger \
one that times out.
- When `verbose=True`, print one line per epoch to stdout (epoch number, training loss, valid \
GAUC/nDCG@5/primary, and the epoch's wall time) — this is streamed live to the user's terminal \
and also parsed back into your own future REFLECTION prompts as a training-curve summary, so keep \
it to one concise line per epoch, not more.
- Don't try to change evaluate.py's scoring semantics — you won't be shown that file and \
can't edit it anyway.
- Only two files are ever applied from a hypothesis turn: data.py and baseline.py (a probe turn \
applies only probe.py — see below). Anything else you write is ignored.
- One attributable change per candidate: primarily touch ONE of data.py or baseline.py. You may \
touch the other file too only if strictly necessary to integrate that one change (e.g. widening a \
field count in data.py may require no baseline.py change at all, since encode() already reports \
`dim` generically) — don't make two independent, unrelated changes in the same candidate, since we \
need to be able to tell what caused any metric change. This applies per candidate even if you're \
racing several in parallel this turn (see the VARIANT format below) — each variant individually \
must still be one attributable change; the variants should differ from EACH OTHER, not each be a \
grab-bag of multiple changes internally.
- Optionally, sweep ONE numeric knob instead of committing to a single value: if your change reads \
a hyperparameter via `hparams.get(name, default)` (an existing one, or one you're introducing this \
turn for something currently hardcoded — e.g. MLP width, dropout, k, lr, epochs, weight decay), you \
may name it in SWEEP_PARAM and give 2-4 candidate values in SWEEP_VALUES (see output format below). \
The harness then runs your one code change once per value — same code, no extra LLM calls — and \
scores the node on whichever value did best. This is still one attributable change (the knob you \
picked), just with its value chosen empirically instead of guessed once; don't combine it with also \
changing something structurally unrelated in the same turn.
- Never use a same-row post-impression outcome field as a raw input feature: play_time_ms, \
profile_stay_time, comment_stay_time, is_click, is_like, is_follow, is_comment, is_forward, \
is_hate, is_profile_enter. These are recorded concurrently with the row's own label and \
wouldn't be known yet at serving time in a real system — using them directly is a label leak, \
not a real feature. An *aggregated historical* version of one of these (e.g. a user's \
long_view rate over their own prior rows, computed so it never looks at the current row) is \
fine and is a legitimate feature idea.
- If a retrieved literature note or a diagnostic probe finding below directly informed your \
hypothesis, name it (e.g. "per ESMM..." or "per the probe on node 7...") in the HYPOTHESIS \
line — this is tracked for the project's write-up. Don't force a citation where none applies."""

_PROBE_CONTRACT = """If you choose to write a probe instead of a hypothesis this turn: you do not \
have file access. You are given, as plain function arguments, the exact same pre-loaded objects \
agent/eda.py's own analysis uses internally:
- `arrays`: dict[str, np.ndarray] — one int64 array per raw log column (user_id, video_id, date, \
hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, \
play_time_ms, duration_ms, profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab), \
each aligned row-for-row across both log files concatenated.
- `masks`: dict['train' | 'valid' | 'test' -> boolean np.ndarray] selecting each split's rows (same \
length as every array in `arrays`).
- `user_feat` / `video_feat`: dict[id_str -> dict[column_name -> string_value]], parsed straight from \
user_features_pure.csv / video_features_basic_pure.csv — every value is a raw string, cast as needed \
(some numeric-looking columns carry sentinel values — see the EDA summary below).
- `label`: the label column name (str) — currently 'long_view'. Index into `arrays[label]`.

Write exactly one function, named exactly `run_probe`, in a file named exactly `probe.py`:
```python:probe.py
def run_probe(arrays, masks, user_feat, video_feat, label):
    ...
    return {...}  # must be a JSON-serializable dict of your findings
```
Constraints: numpy/stdlib only, no file I/O, no network access, read-only (don't mutate the \
arguments), same leakage rule as above (a same-row post-impression field isn't a legitimate signal \
to explain the label with), fast (well under a minute on CPU — compute exactly what answers the \
question), and return a small number of named statistics, not raw arrays or large tables, since \
this gets pasted verbatim into future prompts."""

_OUTPUT_FORMAT = f"""Output format, exactly one of the following seven:

1) Propose a hypothesis and code change:

REFLECTION: <1-3 sentences reflecting on your last iteration (shown to you below, under "Your \
last iteration") -- did the outcome match what you expected, and what's the most likely \
explanation either way? If a training curve is shown, use it (e.g. "it overfit -- validation \
peaked at epoch 4 then degraded" is a real diagnosis; "it just didn't work" is not). Omit this \
line entirely if this is your first iteration.>

HYPOTHESIS: <1-3 sentences: what you're changing (a feature, the model architecture, the \
training/loss procedure -- whichever you judge highest-leverage right now) and why you think it \
will help>
EXPECTED_EFFECT: <1 sentence: which metric (GAUC vs nDCG@5) you expect to move more, and why -- \
this is checked against the actual result and referenced in your next REFLECTION, so commit to a \
real, falsifiable prediction rather than a hedge>
LEVER_CATEGORY: <exactly one category name from the taxonomy above, e.g. "popularity_debiasing" -- \
required, not optional>
SWEEP_PARAM: <optional -- omit entirely unless you want to sweep. The exact hparams.get(...) key \
your code below reads for the one knob you want to try several values of>
SWEEP_VALUES: <optional, required if SWEEP_PARAM is given -- 2 to 4 comma-separated numeric values, \
e.g. "32, 64, 128">

```python:data.py
<full new file content, only if you are changing this file>
```

```python:baseline.py
<full new file content, only if you are changing this file>
```

(omit whichever file's block you're not changing)

2) OR, instead, investigate first by writing a read-only diagnostic probe (only if you genuinely \
don't have enough information to propose a good hypothesis right now -- e.g. you're unsure \
whether a mechanism you'd like to try is actually supported by the data):

REFLECTION: <same as above>

PROBE_QUESTION: <a single, specific, answerable question about the raw data that a short numpy \
computation could resolve, which nothing computed so far (EDA report, prior probes) has already \
answered>

```python:probe.py
<full file content per the probe contract below>
```

3) OR, instead, ask a live web-research question (only if you suspect a specific published \
method/technique would help but the literature/EDA/probe context below doesn't already cover it \
well enough to act on -- check that context first; don't ask about something already answered \
there, and don't use this for a question about THIS project's own data, that's what a probe is \
for). Only available if the budget line below shows at least one remaining -- if it shows zero, \
don't choose this option, propose a hypothesis or probe instead:

REFLECTION: <same as above>

RESEARCH_QUESTION: <a single, specific question about a published method/technique -- e.g. "is \
there a lightweight way to debias implicit-feedback popularity skew that doesn't require a \
randomized-exposure log?" -- answerable by searching the web, not by computing something over this \
project's own data>

4) OR, instead, ask an EXPLORE_QUESTION about THIS repository itself (only if you suspect an \
existing script, prior write-up, or unused reference implementation in this codebase already \
answers something you're about to guess at -- check the EDA/literature/probe context below first; \
don't use this for a question about the raw data, that's a probe, or about a published external \
method, that's RESEARCH_QUESTION). Only available if the budget line below shows at least one \
remaining:

REFLECTION: <same as above>

EXPLORE_QUESTION: <a single, specific question about what already exists in this project's own \
repo -- e.g. "is there already a script comparing how many encoded fields the FM baseline uses?" \
-- answerable by reading this codebase, not by computing something over the raw data or searching \
the web>

5) OR, instead, request a new EDA_ROUND_REQUEST (only if you suspect recent lack of progress is a \
*data-understanding* gap rather than a *modeling* gap -- e.g. you're plateaued, or your last few \
EXPECTED_EFFECT predictions have been wrong in a way that suggests you're missing something about \
the data itself, not just picking weak mechanisms. This triggers a fresh, genuinely exploratory \
round of read-only data probing -- distinct from option 2 above, which answers one question you \
already know to ask; this instead lets a separate pass look around and decide for itself what's \
worth checking, then folds whatever it finds into the EDA summary below for good, so don't use this \
for a question you can already state precisely -- that's a probe. Only available if the budget line \
below shows at least one remaining:

REFLECTION: <same as above>

EDA_ROUND_REQUEST: <1-2 sentences: why you suspect a data-understanding gap right now, tying it to \
what just happened (a plateau, a wrong prediction, an unexpected result) -- this is given to the \
exploration round as its starting focus>

6) OR, instead of guessing a full replacement file body in one shot, request a CODE_SESSION_REQUEST \
-- a bounded multi-turn session that can read the current data.py/baseline.py and edit them \
directly, iterating before handing off (rather than you writing a whole file blind and only \
finding out if it runs via the repair loop after). Worth it mainly for a change gnarly/unfamiliar \
enough that reading-while-editing genuinely helps -- not the default choice for an easily-specified \
small change, which is still cheaper and just as effective as a normal hypothesis turn. Only \
available if the budget line below shows at least one remaining:

REFLECTION: <same as above>

CODE_SESSION_REQUEST: <1-2 sentences: what you want changed and why a multi-turn read-edit session \
is worth it for this particular change -- this is given to the session as its starting context>

7) OR, instead of committing to one hypothesis, race 2 to {MAX_VARIANTS} DIFFERENT hypotheses in \
parallel this turn -- worth doing when you have several genuinely distinct, comparably promising \
ideas and aren't sure which is best, especially when plateaued and unsure whether the right lever is \
a feature, the model, or the training procedure. The harness runs every variant once at default \
hyperparameters, keeps whichever beats current best by the most (or none, if none do), and feeds \
every variant's hypothesis + outcome -- including the losing ones -- into your next iteration's \
context. Each variant must still be ONE attributable change (see the contract above) and genuinely \
different from the others -- don't make variant 2 a small tweak of variant 1. A variant MAY also \
carry its own optional SWEEP_PARAM/SWEEP_VALUES (same rules as option 1) -- but hyperparameter \
sweeping is a refinement of an already-chosen mechanism, not a way to pick between mechanisms, so \
it's only ever actually run on whichever ONE variant wins the initial race; a losing variant's sweep \
spec is simply discarded unexecuted, so don't hesitate to add one to any variant you think might \
need it if it wins:

REFLECTION: <same as above, once, before the first variant -- not repeated per variant>

VARIANT 1:
HYPOTHESIS: <as in option 1>
EXPECTED_EFFECT: <as in option 1>
LEVER_CATEGORY: <required, as in option 1 -- ideally each variant is a DIFFERENT category, since \
racing several mechanisms from different untried categories is exactly what this option is for>
SWEEP_PARAM: <optional, same as option 1>
SWEEP_VALUES: <optional, required if SWEEP_PARAM is given>
```python:data.py
<only if this variant changes it>
```
```python:baseline.py
<only if this variant changes it>
```

VARIANT 2:
HYPOTHESIS: <a genuinely different mechanism from variant 1>
EXPECTED_EFFECT: <as in option 1>
LEVER_CATEGORY: <required>
```python:baseline.py
<full file content for this variant>
```

(up to {MAX_VARIANTS} variants total; each variant's fenced blocks are independent full-file \
rewrites starting from the CURRENT best code shown below, not diffs against each other)

Output nothing else outside one of these seven formats — no preamble, no explanation after the code, \
and never combine HYPOTHESIS, PROBE_QUESTION, RESEARCH_QUESTION, EXPLORE_QUESTION, EDA_ROUND_REQUEST, \
CODE_SESSION_REQUEST, or VARIANT blocks in the same response (a multi-variant response uses ONLY \
VARIANT blocks, no top-level HYPOTHESIS/PROBE_QUESTION/RESEARCH_QUESTION/EXPLORE_QUESTION/ \
EDA_ROUND_REQUEST/CODE_SESSION_REQUEST line)."""


def build_system_prompt(torch_available=True):
    """torch_available comes from orchestrator.py's startup preflight (v4 roadmap Phase 3a):
    checked once in the exact interpreter agent/run_and_report.py's subprocess will use, so the
    propose/repair role never asserts an installed-package claim that isn't actually true this
    run -- see _build_contract()/_TORCH_UNAVAILABLE_LINE above. Recomputed per call (not cached)
    since the model-zoo reference and contract text are cheap to rebuild and torch_available can
    legitimately differ across calls only in tests, never within one real orchestrator run."""
    return f"""You are an ML engineering agent iterating on a recommendation-ranking baseline \
(KuaiRand-Pure). Each turn, look at the CURRENT full pipeline — data.py's feature engineering AND \
baseline.py's model/training procedure together, not a pre-assigned axis — and decide where the \
single highest-leverage next move is: a feature/preprocessing change, a model/architecture change, \
or a training-procedure/loss change. You are free to pick whichever kind of change seems most \
promising this turn based on the reflection, EDA facts, literature, and probe findings below; \
nothing restricts you to repeating the same kind of change turn after turn, but also nothing \
requires variety for its own sake — a genuinely good feature idea two turns in a row is fine, a \
weak variant of one that already failed is not (see the plateau warning below, when present). A \
rejected idea from earlier isn't necessarily dead, either — it was only tested against whatever the \
best pipeline looked like at that point; if the best pipeline has since changed in a way that plausibly \
changes the mechanism (e.g. a new architecture with more capacity, or a new loss), a specific, \
reasoned case for retrying an old idea is legitimate. This is an option to weigh, not an obligation — \
don't retry old ideas by default or just for coverage.

{_build_contract(torch_available)}

{_PROBE_CONTRACT}

{_LEVER_TAXONOMY}

Your job each turn: propose exactly ONE focused hypothesis for improving the validation `primary` \
metric (mean of GAUC and nDCG@5) — via a feature, the model, or training/loss — and rewrite the \
full contents of whichever single file that requires; or, if you lack the information to do that \
well, write a probe, ask a research question, or ask an explore question instead (a probe answers \
a numeric question about the raw data, a research question answers what published methods say, an \
explore question answers what's already sitting in this project's own repo — pick whichever \
matches what you're actually missing); or, if you have several distinct, comparably promising \
ideas and aren't sure which is best, race them as parallel variants (see option 5 below) rather \
than guessing which one to commit to. Whichever you choose, keep it to one attributable change per \
candidate, not a grab-bag of unrelated changes, since we need to be able to tell what caused any \
metric change.

{_OUTPUT_FORMAT}

Reference model zoo (only relevant if you're proposing a model/architecture change — \
field_dims/embed_dim-style constructors, takes a LongTensor of shape (batch, num_fields)):

{_load_model_zoo_reference()}"""


def build_code_session_system_prompt(torch_available=True):
    """v4 roadmap Phase 4. Unlike build_system_prompt() above, this role has real Read/Grep/
    Glob/Edit/Write tools (via llm.call_claude_code(), cwd-confined to one candidate node
    directory -- see that function's docstring for why Bash is deliberately never granted
    here) and runs as its own bounded multi-turn `claude -p` session, not a single text
    completion. It shares the exact same rules (_build_contract) and lever taxonomy as the
    plain propose role -- the only real difference is HOW the change gets made (reading and
    editing files directly instead of guessing a whole replacement body in one shot) and
    that its final hand-off is metadata-only text, since the file changes are already on
    disk via its own tool calls by the time it finishes."""
    return f"""You are an ML engineering agent making ONE focused, attributable change to a \
recommendation-ranking baseline (KuaiRand-Pure), working directly in a candidate directory that \
already contains the current best `data.py` and `baseline.py`.

You have Read/Grep/Glob/Edit/Write tools, confined to your working directory — you do NOT have \
Bash, PowerShell, or any other code-execution/network tool, and you cannot reach any file outside \
this directory. Nothing you edit is scored by you; a separate, fixed harness runs your finished \
candidate afterward exactly the way it runs every other candidate.

Workflow: Read `data.py` and `baseline.py` first (don't guess their current contents from the \
summary below). Make your one change directly with Edit/Write. Re-read whatever you touched if \
you want to double-check it before finishing — you have several turns available, not just one \
shot. When you are done, your FINAL message must be exactly this and nothing else — no fenced \
code blocks (your edits are already saved to disk, that IS the change):

REFLECTION: <1-3 sentences, same as a normal propose turn — what you learned from the run context \
below and what you did>
HYPOTHESIS: <the one change you made, stated as a hypothesis about why it should help>
EXPECTED_EFFECT: <1 sentence: the metric effect you expect and why>
LEVER_CATEGORY: <exactly one of the categories below>

{_build_contract(torch_available)}

{_LEVER_TAXONOMY}

Why this session was requested (the propose role's own stated reason — treat it as your starting \
brief, not a rigid spec) and the same grounding context a normal propose turn gets (EDA facts, \
literature, probe findings, run history) are given to you below. Sweeping a hyperparameter \
(SWEEP_PARAM/SWEEP_VALUES) is not supported in this mode — commit to one value."""


def build_code_session_prompt(best_code, history, context_note):
    """User-prompt for a code session -- reuses the same private context-loader helpers
    build_propose_prompt() uses (EDA/literature/probes/history) rather than duplicating
    them, so the two prompts never drift out of sync on what "current context" means.
    best_code is shown for reference only (the session's actual working copy is what's on
    disk in its cwd, which is what it must Read and Edit -- this is just so the model
    doesn't have to Read before it can start reasoning about what to change)."""
    eda_txt = _load_eda_summary()
    lit_txt = _load_literature_context()
    probe_txt = _load_probe_findings()
    hist_txt = _format_history(history)
    return f"""Why this session was requested: {context_note or '(no reason given)'}

Data facts from EDA (computed once, directly from the real CSVs -- treat as ground truth):
{eda_txt}

Findings from targeted diagnostic probes run so far (also ground truth):
{probe_txt}

Relevant published methods (retrieved from a local corpus):
{lit_txt}

History of past iterations (for context — avoid repeating what's already been tried):
{hist_txt}

For reference, the current data.py and baseline.py (read the actual files in your working \
directory before editing them — this is only so you don't have to Read before starting to think \
about what to change):
```python
{best_code['data.py']}
```
```python
{best_code['baseline.py']}
```

Begin: read the files in your working directory, make your one change, then finish with the \
REFLECTION/HYPOTHESIS/EXPECTED_EFFECT/LEVER_CATEGORY format from your instructions."""


def _format_sweep(h):
    sweep = h.get('sweep')
    if not sweep:
        return ""
    parts = []
    for t in sweep['trials']:
        mark = '*' if sweep.get('best_value') == t['value'] and t['ok'] else ''
        parts.append(f"{t['value']}→{t['primary']:.4f}{mark}" if t['ok'] else f"{t['value']}→FAILED")
    return f" [swept {sweep['param']}: {', '.join(parts)}]"


def _format_variants(h):
    """Renders every variant's hypothesis + outcome from a parallel_hypothesis node --
    including the losing ones, per the user's explicit request that a losing variant's
    insight should still reach the next iteration, not just the winner's."""
    bits = []
    for vr in h['variants']:
        mark = ' [WINNER]' if vr['variant'] == h.get('winning_variant') else ''
        cat = f" ({vr['lever_category']})" if vr.get('lever_category') else ''
        if vr['ok']:
            outcome = f"valid primary {vr['metrics']['valid']['primary']:.4f}"
        else:
            outcome = f"FAILED ({(vr.get('error') or '')[:80]})"
        bits.append(f"    v{vr['variant']}{mark}{cat}: \"{vr['hypothesis'][:100]}\" -> {outcome}")
    return "\n" + "\n".join(bits)


HISTORY_FULL_DETAIL_WINDOW = 8


def _trunc(s, n=70):
    s = (s or '').replace('\n', ' ').strip()
    return s if len(s) <= n else s[:n - 1] + '…'


def _format_history_entry(h):
    """Full-detail rendering for one history entry -- unchanged from before windowing
    was added, just factored out of _format_history() so it can be reused for whichever
    entries fall inside the full-detail window."""
    status = h['status']
    if h.get('variants'):
        if status == 'accepted':
            tag = 'ACCEPTED' if h.get('significant') else 'ACCEPTED (insignificant, below epsilon)'
        else:
            tag = 'rejected (no improvement)'
        swept_note = ', winner then swept' if h.get('sweep') else ''
        return (f"iter {h['iter']}: {tag} (raced {len(h['variants'])} parallel variants"
                f"{swept_note}){_format_sweep(h)} "
                f"-> best primary {h['primary']:.4f} "
                f"({'was' if status == 'accepted' else 'best stays'} {h['prev_best']:.4f})"
                f"{_format_variants(h)}")
    if status == 'accepted':
        cat = f" [{h['lever_category']}]" if h.get('lever_category') else ''
        session_note = ' (via multi-turn coding session)' if h.get('authored_via') == 'code_session' else ''
        sig_note = '' if h.get('significant') else ' (insignificant, below epsilon)'
        return (f"iter {h['iter']}: ACCEPTED{cat}{sig_note} \"{h['hypothesis']}\"{_format_sweep(h)}{session_note} "
                f"-> primary {h['primary']:.4f} (was {h['prev_best']:.4f})")
    if status == 'rejected':
        cat = f" [{h['lever_category']}]" if h.get('lever_category') else ''
        session_note = ' (via multi-turn coding session)' if h.get('authored_via') == 'code_session' else ''
        return (f"iter {h['iter']}: rejected{cat} (no improvement) \"{h['hypothesis']}\""
                f"{_format_sweep(h)}{session_note} -> primary {h['primary']:.4f} (best stays {h['prev_best']:.4f})")
    if status == 'answered':
        return f"iter {h['iter']}: PROBE \"{h.get('question', '')}\" -> see diagnostic probe findings below"
    if status == 'failed' and h.get('question') is not None:
        return f"iter {h['iter']}: probe FAILED \"{h.get('question', '')}\""
    if status == 'researched':
        return (f"iter {h['iter']}: WEB RESEARCH \"{h.get('research_question', '')}\" "
                f"-> found \"{h.get('note_title', '')}\", see literature above")
    if status == 'research_failed':
        return (f"iter {h['iter']}: web research FAILED (no reliable source found) "
                f"\"{h.get('research_question', '')}\"")
    if status == 'research_denied':
        return (f"iter {h['iter']}: web research budget was already exhausted, request "
                f"denied \"{h.get('research_question', '')}\"")
    if status == 'explored':
        return (f"iter {h['iter']}: REPO EXPLORE \"{h.get('explore_question', '')}\" "
                f"-> found \"{h.get('note_title', '')}\", see literature above")
    if status == 'explore_failed':
        return (f"iter {h['iter']}: repo explore FAILED (nothing relevant found) "
                f"\"{h.get('explore_question', '')}\"")
    if status == 'explore_denied':
        return (f"iter {h['iter']}: explore budget was already exhausted, request "
                f"denied \"{h.get('explore_question', '')}\"")
    if status == 'eda_round':
        return (f"iter {h['iter']}: EDA ROUND \"{h.get('eda_round_question', '')}\" "
                f"-> see updated EDA summary above")
    if status == 'eda_round_failed':
        return (f"iter {h['iter']}: EDA round FAILED (exploration loop found nothing new) "
                f"\"{h.get('eda_round_question', '')}\"")
    if status == 'eda_round_denied':
        return (f"iter {h['iter']}: EDA round budget was already exhausted, request "
                f"denied \"{h.get('eda_round_question', '')}\"")
    if status == 'code_session_denied':
        return (f"iter {h['iter']}: code session budget was already exhausted, request "
                f"denied \"{h.get('code_session_question', '')}\"")
    return f"iter {h['iter']}: FAILED after retries \"{h.get('hypothesis')}\" -> {h.get('error_summary')}"


def _format_history_compact(h):
    """One-line summary for an entry that's aged out of the full-detail window (added
    after a real run's propose prompt grew to ~18K input tokens by node 14, purely from
    every past iteration's full hypothesis/reflection text accumulating forever -- an
    unbounded per-call cost that both risks LLM-call timeouts on generation and directly
    inflates the token-consumption metric the hackathon scores under Feasibility &
    Practicality. Nothing is lost, only compressed: the full record stays in
    logs/node_N.json regardless of what's shown here.)."""
    status = h['status']
    cat = f" [{h['lever_category']}]" if h.get('lever_category') else ''
    if status in ('accepted', 'rejected'):
        raced = f" (raced {len(h['variants'])})" if h.get('variants') else ''
        session_note = ' (via code session)' if h.get('authored_via') == 'code_session' else ''
        sig_note = '' if (status != 'accepted' or h.get('significant')) else ' (insignificant)'
        return (f"iter {h['iter']}: {status}{sig_note}{cat}{raced}{session_note} "
                f"\"{_trunc(h.get('hypothesis'))}\" -> {h['primary']:.4f}")
    if status == 'answered':
        return f"iter {h['iter']}: probe -> \"{_trunc(h.get('question'))}\""
    if status == 'failed' and h.get('question') is not None:
        return f"iter {h['iter']}: probe FAILED -> \"{_trunc(h.get('question'))}\""
    if status == 'researched':
        return f"iter {h['iter']}: web research -> found \"{_trunc(h.get('note_title'))}\""
    if status == 'research_failed':
        return f"iter {h['iter']}: web research -> no reliable source found"
    if status == 'research_denied':
        return f"iter {h['iter']}: web research -> denied (budget exhausted)"
    if status == 'explored':
        return f"iter {h['iter']}: repo explore -> found \"{_trunc(h.get('note_title'))}\""
    if status == 'explore_failed':
        return f"iter {h['iter']}: repo explore -> nothing relevant found"
    if status == 'explore_denied':
        return f"iter {h['iter']}: repo explore -> denied (budget exhausted)"
    if status == 'eda_round':
        return f"iter {h['iter']}: EDA round -> \"{_trunc(h.get('eda_round_question'))}\""
    if status == 'eda_round_failed':
        return f"iter {h['iter']}: EDA round -> found nothing new"
    if status == 'eda_round_denied':
        return f"iter {h['iter']}: EDA round -> denied (budget exhausted)"
    if status == 'code_session_denied':
        return f"iter {h['iter']}: code session -> denied (budget exhausted)"
    return f"iter {h['iter']}: FAILED{cat} -> \"{_trunc(h.get('hypothesis'))}\""


def _format_history(history, full_window=HISTORY_FULL_DETAIL_WINDOW):
    if not history:
        return "(no iterations yet)"
    older, recent = history[:-full_window], history[-full_window:]
    lines = []
    if older:
        lines.append(f"(the {len(older)} earliest iteration(s) are compressed to one line each below to "
                     f"bound prompt size -- full detail is kept for the most recent {len(recent)} and in "
                     f"logs/node_N.json regardless; nothing has been forgotten, just summarized)")
        lines.extend(_format_history_compact(h) for h in older)
    lines.extend(_format_history_entry(h) for h in recent)
    return "\n".join(lines)


def _format_curve(curve):
    if not curve:
        return ""
    parts = [f"Training curve: {curve['n_epochs_logged']} epochs logged, peaked at epoch "
             f"{curve['best_epoch']} (primary {curve['best_epoch_primary']:.4f}), ended at epoch "
             f"{curve['last_epoch']} (primary {curve['last_epoch_primary']:.4f})"]
    if curve.get('degraded_after_best'):
        parts.append("— degraded after peaking (an overfitting signature: consider more "
                     "regularization/less capacity, not necessarily a bad underlying idea).")
    elif curve.get('still_improving_at_cutoff'):
        parts.append("— still improving when it stopped without early-stopping (possibly "
                     "underfit: more epochs/patience might help before concluding this idea failed).")
    return " ".join(parts)


def _format_reflection_block(history):
    """Reflects on the most recent hypothesis/probe attempt (skips nothing -- unlike v1 there's
    only one stream now, so this is just "the last entry with an outcome")."""
    last = None
    for h in reversed(history):
        if h.get('status') in ('accepted', 'rejected', 'failed', 'answered',
                                'researched', 'research_failed', 'research_denied',
                                'explored', 'explore_failed', 'explore_denied',
                                'eda_round', 'eda_round_failed', 'eda_round_denied'):
            last = h
            break
    if last is None:
        return "(this is your first iteration — nothing to reflect on yet.)"
    if last['status'] == 'answered':
        return (f"Your last iteration ran a diagnostic probe rather than proposing a change: "
                f"\"{last.get('question')}\" — see the findings below for the answer.")
    if last['status'] == 'researched':
        return (f"Your last iteration asked a web-research question rather than proposing a "
                f"change: \"{last.get('research_question')}\" — found \"{last.get('note_title')}\", "
                f"see the literature section below.")
    if last['status'] in ('research_failed', 'research_denied'):
        reason = ('no reliable source was found' if last['status'] == 'research_failed'
                  else 'the web-research budget was already exhausted')
        return (f"Your last iteration asked a web-research question rather than proposing a "
                f"change: \"{last.get('research_question')}\" — but {reason}, so nothing new is "
                f"available; propose a hypothesis or probe based on what's already known instead.")
    if last['status'] == 'explored':
        return (f"Your last iteration asked an explore question rather than proposing a "
                f"change: \"{last.get('explore_question')}\" — found \"{last.get('note_title')}\", "
                f"see the literature section below.")
    if last['status'] in ('explore_failed', 'explore_denied'):
        reason = ('nothing relevant was found in this repo' if last['status'] == 'explore_failed'
                  else 'the explore budget was already exhausted')
        return (f"Your last iteration asked an explore question rather than proposing a "
                f"change: \"{last.get('explore_question')}\" — but {reason}, so nothing new is "
                f"available; propose a hypothesis or probe based on what's already known instead.")
    if last['status'] == 'eda_round':
        return (f"Your last iteration requested a new EDA round rather than proposing a "
                f"change: \"{last.get('eda_round_question')}\" — its findings are already folded "
                f"into the EDA summary above, so use that directly rather than re-asking.")
    if last['status'] in ('eda_round_failed', 'eda_round_denied'):
        reason = ('the exploration loop finalized without finding anything new'
                  if last['status'] == 'eda_round_failed'
                  else 'the EDA round budget was already exhausted')
        return (f"Your last iteration requested a new EDA round rather than proposing a "
                f"change: \"{last.get('eda_round_question')}\" — but {reason}, so nothing new is "
                f"available; propose a hypothesis or probe based on what's already known instead.")
    if last.get('variants'):
        lines = [f"You raced {len(last['variants'])} parallel variants last turn; the winner was: "
                 f"\"{last.get('hypothesis')}\"{_format_sweep(last)}",
                 f"All variants and their outcomes (pre-sweep, at default hyperparameters):"
                 f"{_format_variants(last)}"]
    else:
        lines = [f"Hypothesis: {last.get('hypothesis')}{_format_sweep(last)}"]
    if last.get('expected_effect'):
        lines.append(f"Expected effect (stated at the time): {last['expected_effect']}")
    if last['status'] == 'failed':
        lines.append(f"Result: FAILED after retries -- {last.get('error_summary')}")
        return "\n".join(lines)
    sig_note = '' if (last['status'] != 'accepted' or last.get('significant')) else ' (insignificant, below epsilon)'
    lines.append(f"Result: {last['status']}{sig_note} -- valid primary {last['primary']:.4f} "
                 f"(was {last['prev_best']:.4f})")
    v = last.get('metrics', {}).get('valid', {})
    pv = last.get('prev_metrics', {}).get('valid', {})
    if v and pv:
        lines.append(f"Actual per-metric change: GAUC {pv.get('GAUC'):.4f} -> {v.get('GAUC'):.4f} "
                     f"({v.get('GAUC') - pv.get('GAUC'):+.4f}), nDCG@5 {pv.get('nDCG@5'):.4f} -> "
                     f"{v.get('nDCG@5'):.4f} ({v.get('nDCG@5') - pv.get('nDCG@5'):+.4f})")
    curve_txt = _format_curve(last.get('training_curve'))
    if curve_txt:
        lines.append(curve_txt)
    return "\n".join(lines)


def top_candidates(history, top_k=5):
    """Every attempted node with a real trained-model score (accepted or rejected --
    excludes research/probe/failed turns, which have no score), ranked descending by
    valid primary. Every node here that beat the best-at-the-time was accepted (any
    positive gain replaces best_dir, see orchestrator.py), so 'rejected' now strictly
    means it scored at or below whatever was best when it ran -- still useful
    ensembling material for a genuinely different mechanism sitting outside the
    current chain, just not a "secretly better than best" candidate the way it could
    be before accept/reject and significance (ACCEPT_EPS) were decoupled."""
    candidates = [h for h in history if h.get('status') in ('accepted', 'rejected')
                  and h.get('primary') is not None]
    return sorted(candidates, key=lambda h: -h['primary'])[:top_k]


def _format_top_candidates(history, top_k=5):
    top = top_candidates(history, top_k)
    if not top:
        return "(no scored candidates yet)"
    return "\n".join(
        f"- iter {h['iter']} [{h.get('lever_category') or 'uncategorized'}] "
        f"{h['status']}{'' if (h['status'] != 'accepted' or h.get('significant')) else ' (insignificant)'}: "
        f"valid primary {h['primary']:.4f} -- \"{_trunc(h.get('hypothesis'), 90)}\""
        for h in top)


def _format_ensemble_nudge(history, best_primary, min_count=2):
    """Forceful, STANDING nudge -- deliberately NOT gated on plateau streak the way
    _format_escalation below is. Every other forceful nudge so far (RESEARCH_QUESTION,
    the untried-category requirement) needed a "you've been stuck for a while" story to
    justify itself, which meant node 17 of a fresh invocation (streak resets to 0 on
    resume) would see zero pressure toward ensembling even though the evidence for it
    doesn't need a streak to build up; it's already concretely true right now.
    Historically (see agent_notes discussion), this model has only ever adopted an
    unusual mode when a nudge was forceful and required, never from a passive "you
    could also try X" suggestion -- so this mirrors that same forceful phrasing, just
    triggered by a different, streak-independent condition. Excludes past 'ensembling'
    attempts from the candidate pool -- a prior ensemble is itself already a
    combination, not a fresh single mechanism worth suggesting as a NEW pairing
    partner, and counting its category toward "these span different categories" would
    be misleading (ensembling two single-mechanism candidates isn't diversity in the
    sense that actually helps).

    Trigger condition (reworked when accept/reject and significance were decoupled --
    see orchestrator.py's module docstring): used to fire on *rejected* nodes that
    individually scored above the current best without clearing ACCEPT_EPS -- that
    population is now empty by construction, since any node that beats best gets
    accepted immediately. The equivalent signal now is nodes that WERE accepted but
    only by an insignificant (<=ACCEPT_EPS) margin: each one moved the chain forward a
    little on its own, individually too small to be a strong result, which is exactly
    the situation where combining two decorrelated small-gain mechanisms is worth
    trying. Returns '' if fewer than `min_count` such candidates exist."""
    insignificant_accepts = [h for h in top_candidates(history, top_k=10)
               if h.get('status') == 'accepted' and not h.get('significant')
               and h.get('lever_category') != 'ensembling']
    if len(insignificant_accepts) < min_count:
        return ""
    names = ", ".join(f"iter {h['iter']} ({h['primary']:.4f}, {h.get('lever_category') or 'uncategorized'})"
                       for h in insignificant_accepts)
    distinct_cats = {h.get('lever_category') for h in insignificant_accepts if h.get('lever_category')}
    if len(distinct_cats) < 2:
        diversity_note = (
            " Note: these all share the same lever_category, which is the specific failure mode node 17 "
            "already hit -- it ensembled two candidates from the same mechanism (plain BPR-FM + the same "
            "FM's auxiliary-head variant) and got essentially no gain (0.6039 either way), because "
            "near-identical models make near-identical errors and averaging doesn't cancel those out. If "
            "a candidate with a DIFFERENT lever_category exists further down the full ranked list -- even "
            "at a lower individual score -- pairing with that is more likely to actually help than "
            "combining these two.")
    else:
        diversity_note = " These span different lever_categories, which is what actually makes ensembling likely to help."
    return (f"\n★ {len(insignificant_accepts)} past candidates were each accepted but only by an INSIGNIFICANT margin "
            f"(current best {best_primary:.4f}, each individually cleared best-at-the-time by "
            f"<= epsilon): {names}.{diversity_note} "
            f"This turn, strongly prefer ENSEMBLING two candidates with DIFFERENT lever categories "
            f"(LEVER_CATEGORY: ensembling) over proposing something brand new -- averaging genuinely "
            f"decorrelated models' predictions can reduce enough variance to clear the significance margin "
            f"even when neither does alone, and at least one candidate's full code is given to you below, "
            f"verbatim, specifically so you can reuse it rather than re-deriving it from memory. Only "
            f"propose something else if you have a specific reason ensembling wouldn't work here, stated "
            f"in your REFLECTION.\n")


def _tried_categories(history):
    """Every LEVER_CATEGORY actually attempted so far, regardless of accept/reject
    outcome -- scanned from both plain hypothesis turns (top-level 'lever_category')
    and every variant of a parallel_hypothesis turn (a LOSING variant still counts as
    having tried its category; coverage is what matters here, not just what won).
    'uncategorized' (a parse-failure fallback, never an LLM choice) is excluded."""
    tried = set()
    for h in history:
        cat = h.get('lever_category')
        if cat and cat != 'uncategorized':
            tried.add(cat)
        for vr in (h.get('variants') or []):
            vcat = vr.get('lever_category')
            if vcat and vcat != 'uncategorized':
                tried.add(vcat)
    return tried


def _format_escalation(streak, escalate_after, eps, history, web_research_remaining=0, ensemble_active=False):
    """`ensemble_active` (set by build_propose_prompt when _format_ensemble_nudge fired)
    suppresses this function's OWN primary_push (RESEARCH_QUESTION/PROBE_QUESTION) --
    the ensemble nudge already gave a single, forceful, concrete directive for this
    turn, and showing a SECOND competing "strongly preferred action" here would leave
    the model unsure which one actually takes priority. The category-coverage section
    still applies regardless, since that's about future turns' coverage, not this
    turn's single action.

    The organizer's own MLE-iteration diagram has the reflect+revise step loop back
    into re-inspecting the data, not straight into another feature/model guess. An
    earlier version of this text listed "propose something structurally different"
    FIRST, with a probe as an equal-weight "or" option and RESEARCH_QUESTION not
    mentioned at all -- empirically (across 20 real hypothesis turns over two runs) the
    LLM never once chose a probe or a research question, always defaulting to another
    code-level guess even while plateaued. Per the user's explicit direction: when
    plateaued and web-research budget remains, this now makes RESEARCH_QUESTION the
    strongly preferred default action, not just one option among several.

    A second, separate problem showed up in the run that motivated this: even when the
    LLM DID try something "structurally different" per the old vague instruction, it
    kept cycling through the same two categories (feature_addition,
    training_procedure/hyperparameter_capacity) it already knew how to propose, never
    reaching popularity_debiasing or multi_task even though the literature corpus
    mentions both every turn. "Try something different" doesn't mean "try an untried
    CATEGORY" to a model that's only ever seen itself succeed by tweaking features and
    training knobs. LEVER_CATEGORY tagging (see _tried_categories above) fixes this by
    naming the untried categories explicitly, rather than leaving "different" to the
    model's own (evidently narrow) interpretation."""
    if streak < escalate_after:
        return ""
    recent = [h for h in history if h.get('status') in ('accepted', 'rejected')][-streak:]
    recent_txt = "; ".join(f'"{(h.get("hypothesis") or "")[:90]}"' for h in recent) or "(none)"

    tried = _tried_categories(history)
    untried = [c for c in LEVER_CATEGORIES if c != 'other' and c not in tried]
    if untried:
        category_note = (
            f"Categories NOT tried at all yet: {', '.join(untried)}. Categories already tried without "
            f"a lasting improvement: {', '.join(sorted(tried)) or '(none)'}. If you do propose code this "
            f"turn (rather than research/probe), it MUST use one of the untried categories above, not "
            f"another variant of one that's already failed — if several untried categories seem "
            f"promising, race them against each other as parallel VARIANTs (one category per variant; "
            f"see your instructions) instead of guessing which to try first.")
    else:
        category_note = (
            "Every lever category has now been tried at least once without a lasting improvement — this "
            "is a genuine judgment call, not a coverage gap: pick whichever tried category has the "
            "strongest SPECIFIC reason to work differently given what's changed since it was last tried "
            "(e.g. more model capacity now available, a new feature now present), and state that reason "
            "explicitly in your REFLECTION rather than just repeating the attempt.")

    if ensemble_active:
        primary_push = (
            "You are also in a PLATEAU, but the ★ ensembling nudge above already gives you a concrete, "
            "low-risk directive for this turn — follow that instead of researching or probing right now.")
    elif web_research_remaining > 0:
        primary_push = (
            f"You are in a PLATEAU. Your strongly preferred action this turn is RESEARCH_QUESTION "
            f"({web_research_remaining} left this run) — you have already tried {streak} code-level "
            f"changes without a real improvement, so the highest-leverage move now is finding out "
            f"whether there's a published technique for this exact situation that you don't already "
            f"know about, not guessing again. Only skip it if you have a SPECIFIC, well-justified "
            f"reason RESEARCH_QUESTION wouldn't help right now — and if so, say that reason explicitly "
            f"in your REFLECTION before proposing something else.")
    else:
        primary_push = (
            f"You are in a PLATEAU. Your web-research budget is exhausted, so first ask whether a "
            f"read-only PROBE_QUESTION about the data itself would explain why {streak} code-level "
            f"changes in a row haven't worked, before trying yet another guess.")

    return (f"\n⚠ PLATEAU WARNING: your last {streak} iterations each improved validation primary "
            f"by {eps} or less — you appear to be stuck reapplying variants of the same underlying "
            f"mechanism. Recent hypotheses: {recent_txt}. Do NOT propose another small variant of the "
            f"same mechanism as those (e.g. another 'smoothed historical rate for a different entity' "
            f"if that's the pattern above).\n\n{primary_push}\n\nIf research/probing genuinely doesn't "
            f"apply and you're proposing code this turn: {category_note}\n")


def build_propose_prompt(best_code, history, best_primary, streak, escalate_after, converge_eps,
                          web_research_remaining=0, explore_remaining=0, eda_round_remaining=0,
                          code_session_remaining=0, alt_node_id=None, alt_code=None):
    hist_txt = _format_history(history)
    eda_txt = _load_eda_summary()
    lit_txt = _load_literature_context()
    probe_txt = _load_probe_findings()
    reflect_txt = _format_reflection_block(history)
    ensemble_nudge_txt = _format_ensemble_nudge(history, best_primary)
    escalate_txt = _format_escalation(streak, escalate_after, converge_eps, history, web_research_remaining,
                                       ensemble_active=bool(ensemble_nudge_txt))
    top_candidates_txt = _format_top_candidates(history)
    alt_section = ""
    if alt_code:
        alt_section = f"""
Full code for the highest-scoring PAST candidate that ISN'T the current best (iter {alt_node_id}, \
see its score in the ranked list above) -- reuse this VERBATIM if you want to ensemble with it \
(e.g. average its predict() output with the current best's), rather than re-deriving the mechanism \
from memory, which risks subtly reimplementing it differently than what was actually validated:
```python
{alt_code['data.py']}
```
```python
{alt_code['baseline.py']}
```
"""
    return f"""Your last iteration (reflect on this before deciding your next move):
{reflect_txt}
{escalate_txt}
Data facts from EDA (computed once, directly from the real CSVs -- treat as ground truth, not \
something to re-derive from common sense):
{eda_txt}

Findings from targeted diagnostic probes run so far (if any -- these answer specific questions \
raised along the way, and are also ground truth):
{probe_txt}

Relevant published methods (retrieved from a local corpus -- [curated] entries were vetted once by \
a human before this project started, [found via live web search this run] entries were found and \
citation-checked by a separate research call earlier in this run; selected based on the EDA \
findings above -- not an exhaustive literature review, just what's most likely relevant here):
{lit_txt}

Web-research budget remaining this run: {web_research_remaining} (see the RESEARCH_QUESTION option \
in your instructions -- only choose it if this is at least 1, and only for a published-methods \
question the literature above doesn't already answer).

Repo-explore budget remaining this run: {explore_remaining} (see the EXPLORE_QUESTION option in \
your instructions -- only choose it if this is at least 1, and only for a question about what \
already exists in THIS codebase that the context above doesn't already answer).

EDA-round budget remaining this run: {eda_round_remaining} (see the EDA_ROUND_REQUEST option in \
your instructions -- only choose it if this is at least 1, and only when you suspect a genuine \
data-understanding gap, not just to double-check something the EDA facts above already cover).

Code-session budget remaining this run: {code_session_remaining} (see the CODE_SESSION_REQUEST \
option in your instructions -- only choose it if this is at least 1, and only for a change gnarly \
enough that reading-while-editing genuinely helps over guessing a full file body in one shot).

Current best validation primary metric: {best_primary:.4f}

Past candidates ranked by their own valid primary (regardless of accept/reject -- some rejected \
candidates scored close to or above the current best individually without clearing the acceptance \
margin; averaging two independent ones can reduce variance enough to cross it even when neither does \
alone -- see the 'ensembling' lever category if one of these looks worth combining with another):
{top_candidates_txt}
{ensemble_nudge_txt}
{alt_section}
History of past iterations (for context — avoid repeating what's already been tried):
{hist_txt}

Current data.py:
```python
{best_code['data.py']}
```

Current baseline.py:
```python
{best_code['baseline.py']}
```

Decide your next action now, following the output format from the system prompt."""


def build_repair_prompt(attempt_code, hypothesis, error_text):
    files_txt = "\n\n".join(
        f"Current {fname}:\n```python\n{content}\n```"
        for fname, content in attempt_code.items()
    )
    return f"""Your previous attempt for this hypothesis (or probe question) failed to run.

HYPOTHESIS (or PROBE_QUESTION): {hypothesis}

Error output:
```
{error_text[-4000:]}
```

{files_txt}

Fix the bug. Keep the same hypothesis/question. Output ONLY the fenced code block(s) for whichever \
file(s) you're fixing, in the exact format from the system prompt — no REFLECTION, HYPOTHESIS, \
EXPECTED_EFFECT, or PROBE_QUESTION line needed for a repair."""


_FENCE_RE = re.compile(r"```python:([^\n`]+)\n(.*?)```", re.DOTALL)


def parse_fenced_files(text):
    """Extracts every ```python:<filename>\\n...``` fenced block into {filename: content},
    stripping one trailing newline per block. Shared by parse_response(),
    _parse_variant_chunk(), and eda_agent.py's own turn parser."""
    files = {}
    for m in _FENCE_RE.finditer(text):
        fname = m.group(1).strip()
        content = m.group(2)
        if content.endswith("\n"):
            content = content[:-1]
        files[fname] = content
    return files

_REFLECTION_RE = re.compile(
    r"REFLECTION:\s*(.*?)(?:\n\s*(?:HYPOTHESIS:|PROBE_QUESTION:|RESEARCH_QUESTION:|EXPLORE_QUESTION:|"
    r"EDA_ROUND_REQUEST:|CODE_SESSION_REQUEST:|VARIANT\s+\d+:)|\Z)",
    re.DOTALL)  # RESEARCH_QUESTION/EXPLORE_QUESTION/EDA_ROUND_REQUEST/CODE_SESSION_REQUEST/VARIANT
                # stops were missing before this fix -- reflection text could previously run on
                # and swallow a following line (CODE_SESSION_REQUEST added for v4 roadmap Phase 4,
                # same bug class as the others -- see _NEXT_LABEL below)
_HYPOTHESIS_RE = re.compile(r"HYPOTHESIS:\s*(.*?)(?:\n\s*EXPECTED_EFFECT:|\n```|\Z)", re.DOTALL)
_EXPECTED_EFFECT_RE = re.compile(
    r"EXPECTED_EFFECT:\s*(.*?)(?:\n\s*(?:LEVER_CATEGORY:|SWEEP_PARAM:)|\n```|\Z)", re.DOTALL)
_LEVER_CATEGORY_RE = re.compile(r"LEVER_CATEGORY:\s*(.*?)(?:\n\s*SWEEP_PARAM:|\n```|\Z)", re.DOTALL)
# Each of these three is meant to be the ONLY question-type label in a well-formed response
# (see the "never combine" instruction in _OUTPUT_FORMAT), but a malformed response combining
# two of them used to bleed one capture into the next label's text entirely (no stop condition
# but a code fence or end-of-string) -- now each stops at any OTHER label too, so a garbled
# response at least yields two separately-truncated (if useless) captures instead of one
# swallowing the rest of the text.
_NEXT_LABEL = (r"(?:\n\s*(?:HYPOTHESIS:|PROBE_QUESTION:|RESEARCH_QUESTION:|EXPLORE_QUESTION:|"
               r"EDA_ROUND_REQUEST:|CODE_SESSION_REQUEST:|VARIANT\s+\d+:)|\n```|\Z)")
_PROBE_QUESTION_RE = re.compile(r"PROBE_QUESTION:\s*(.*?)" + _NEXT_LABEL, re.DOTALL)
_RESEARCH_QUESTION_RE = re.compile(r"RESEARCH_QUESTION:\s*(.*?)" + _NEXT_LABEL, re.DOTALL)
_EXPLORE_QUESTION_RE = re.compile(r"EXPLORE_QUESTION:\s*(.*?)" + _NEXT_LABEL, re.DOTALL)
_EDA_ROUND_REQUEST_RE = re.compile(r"EDA_ROUND_REQUEST:\s*(.*?)" + _NEXT_LABEL, re.DOTALL)
_CODE_SESSION_REQUEST_RE = re.compile(r"CODE_SESSION_REQUEST:\s*(.*?)" + _NEXT_LABEL, re.DOTALL)
_SWEEP_PARAM_RE = re.compile(r"SWEEP_PARAM:\s*(.*?)(?:\n\s*SWEEP_VALUES:|\n```|\Z)", re.DOTALL)
_SWEEP_VALUES_RE = re.compile(r"SWEEP_VALUES:\s*(.*?)(?:\n```|\Z)", re.DOTALL)
_VARIANT_HEADER_RE = re.compile(r"\n\s*VARIANT\s+(\d+):\s*\n")


def _split_variants(text):
    """Splits a multi-variant response into per-variant text chunks by `VARIANT N:`
    markers. Returns None if fewer than 2 are found -- that's not a multi-variant
    response, callers fall back to normal single-hypothesis/probe/research parsing."""
    padded = '\n' + text  # so a VARIANT marker at the very start of text still matches
    matches = list(_VARIANT_HEADER_RE.finditer(padded))
    if len(matches) < 2:
        return None
    chunks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(padded)
        chunks.append((int(m.group(1)), padded[start:end]))
    return chunks


def _parse_variant_chunk(chunk):
    """Same field extraction as the top-level hypothesis parsing below, scoped to one
    variant's text -- reuses the same regexes since HYPOTHESIS/EXPECTED_EFFECT/SWEEP_*/
    fenced blocks have the same shape inside a variant as at the top level. A variant's
    SWEEP_PARAM/SWEEP_VALUES is parsed here but deliberately NOT executed for every
    variant -- orchestrator.py only runs the sweep on whichever variant wins the
    initial race at default hyperparameters (per the user's framing: sweeping is a
    refinement step on top of an already-chosen mechanism, not a way to pick between
    mechanisms), so a losing variant's sweep spec is simply discarded, unexecuted."""
    hyp_match = _HYPOTHESIS_RE.search(chunk)
    hypothesis = hyp_match.group(1).strip() if hyp_match else ""
    ee_match = _EXPECTED_EFFECT_RE.search(chunk)
    expected_effect = ee_match.group(1).strip() if ee_match else ""
    lc_match = _LEVER_CATEGORY_RE.search(chunk)
    lever_category = _normalize_lever_category(lc_match.group(1) if lc_match else "")
    sweep_param_match = _SWEEP_PARAM_RE.search(chunk)
    sweep_param = sweep_param_match.group(1).strip() if sweep_param_match else ""
    sweep_values = []
    if sweep_param:
        sweep_values_match = _SWEEP_VALUES_RE.search(chunk)
        if sweep_values_match:
            sweep_values = _parse_sweep_values(sweep_values_match.group(1))
        if not sweep_values:
            sweep_param = ""  # SWEEP_PARAM without any usable values -- treat as no sweep
    files = parse_fenced_files(chunk)
    return {'hypothesis': hypothesis, 'expected_effect': expected_effect,
            'lever_category': lever_category, 'sweep_param': sweep_param,
            'sweep_values': sweep_values, 'files': files}


def _parse_sweep_values(raw):
    values = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        values.append(int(v) if v.is_integer() else v)
    return values[:MAX_SWEEP_VALUES]


def parse_response(text):
    """Returns a dict: {'reflection', 'hypothesis', 'expected_effect', 'probe_question',
    'research_question', 'explore_question', 'eda_round_question',
    'code_session_question', 'sweep_param', 'sweep_values', 'mode', 'files', 'variants'}.
    `mode` is 'parallel_hypothesis' if 2+ `VARIANT N:` blocks parsed with at least one
    pipeline file change each (checked FIRST, before the single-hypothesis check below --
    a multi-variant response's fenced blocks also get picked up by the top-level `files`
    scan, which would otherwise misclassify it as a plain 'hypothesis' turn); 'probe' if
    a probe.py file was produced and neither data.py nor baseline.py was (code-based
    detection, not text-label-based, so it's robust to the model accidentally leaving
    stray HYPOTHESIS/PROBE_QUESTION/RESEARCH_QUESTION/EXPLORE_QUESTION/EDA_ROUND_REQUEST/
    CODE_SESSION_REQUEST text around); 'research' if no files at all were produced but a
    RESEARCH_QUESTION line parsed; 'explore' if no files and no research question but an
    EXPLORE_QUESTION line parsed; 'eda_round' if none of the above but an
    EDA_ROUND_REQUEST line parsed; 'code_session' if none of the above but a
    CODE_SESSION_REQUEST line parsed (checked last among the four text-only questions, so
    a response accidentally containing more than one text label still resolves in this
    list's own presentation order -- research, then explore, then eda_round, then
    code_session); otherwise 'hypothesis' (also the fallback for a malformed response with
    no files and no question of any kind -- orchestrator.py's existing "no valid file
    changes parsed" failure path already handles that case, AND the deliberate reuse this
    function gets from agent/code_agent.py's own multi-turn session: its final text has no
    fenced files and no question label either -- by design, since its file changes are
    already applied via Edit/Write, not emitted as text -- so it falls to this same
    'hypothesis' default, which is exactly the mode orchestrator.py's code_session branch
    wants). For a 'parallel_hypothesis' response, the top-level
    'hypothesis'/'expected_effect'/'files' fields are NOT meaningful (they'll contain a
    jumble of whichever variant's blocks the top-level scan happened to see last) -- use
    'variants' instead, a list of {'hypothesis', 'expected_effect', 'files'} dicts, one
    per variant. `sweep_param` is '' and `sweep_values` is [] unless both SWEEP_PARAM and
    at least one valid numeric SWEEP_VALUES entry were present -- a sweep is only ever
    opt-in, so a normal hypothesis turn (the vast majority) is unaffected by this
    parsing. Every field defaults to '' / {} / [] when absent — expected for a repair
    call (no REFLECTION/HYPOTHESIS/EXPECTED_EFFECT/PROBE_QUESTION/RESEARCH_QUESTION/
    EXPLORE_QUESTION/EDA_ROUND_REQUEST/CODE_SESSION_REQUEST/VARIANT needed there) or a
    first-ever iteration."""
    files = parse_fenced_files(text)

    reflection_match = _REFLECTION_RE.search(text)
    reflection = reflection_match.group(1).strip() if reflection_match else ""
    hyp_match = _HYPOTHESIS_RE.search(text)
    hypothesis = hyp_match.group(1).strip() if hyp_match else ""
    sweep_param_match = _SWEEP_PARAM_RE.search(text)
    sweep_param = sweep_param_match.group(1).strip() if sweep_param_match else ""
    sweep_values = []
    if sweep_param:
        sweep_values_match = _SWEEP_VALUES_RE.search(text)
        if sweep_values_match:
            sweep_values = _parse_sweep_values(sweep_values_match.group(1))
        if not sweep_values:
            sweep_param = ""  # SWEEP_PARAM without any usable values -- treat as no sweep
    ee_match = _EXPECTED_EFFECT_RE.search(text)
    expected_effect = ee_match.group(1).strip() if ee_match else ""
    lc_match = _LEVER_CATEGORY_RE.search(text)
    lever_category = _normalize_lever_category(lc_match.group(1) if lc_match else "")
    pq_match = _PROBE_QUESTION_RE.search(text)
    probe_question = pq_match.group(1).strip() if pq_match else ""
    rq_match = _RESEARCH_QUESTION_RE.search(text)
    research_question = rq_match.group(1).strip() if rq_match else ""
    eq_match = _EXPLORE_QUESTION_RE.search(text)
    explore_question = eq_match.group(1).strip() if eq_match else ""
    erq_match = _EDA_ROUND_REQUEST_RE.search(text)
    eda_round_question = erq_match.group(1).strip() if erq_match else ""
    csq_match = _CODE_SESSION_REQUEST_RE.search(text)
    code_session_question = csq_match.group(1).strip() if csq_match else ""

    variants = []
    variant_chunks = _split_variants(text)
    if variant_chunks:
        for _num, chunk in variant_chunks[:MAX_VARIANTS]:
            v = _parse_variant_chunk(chunk)
            if any(f in ALLOWED_FILES for f in v['files']):
                variants.append(v)

    has_pipeline_files = any(f in ALLOWED_FILES for f in files)
    has_probe_file = 'probe.py' in files
    if len(variants) >= 2:
        mode = 'parallel_hypothesis'
    elif has_pipeline_files:
        mode = 'hypothesis'
    elif has_probe_file:
        mode = 'probe'
    elif research_question:
        mode = 'research'
    elif explore_question:
        mode = 'explore'
    elif eda_round_question:
        mode = 'eda_round'
    elif code_session_question:
        mode = 'code_session'
    else:
        mode = 'hypothesis'

    return {'reflection': reflection, 'hypothesis': hypothesis, 'expected_effect': expected_effect,
            'lever_category': lever_category, 'probe_question': probe_question,
            'research_question': research_question, 'explore_question': explore_question,
            'eda_round_question': eda_round_question, 'code_session_question': code_session_question,
            'sweep_param': sweep_param, 'sweep_values': sweep_values, 'mode': mode,
            'files': files, 'variants': variants}
