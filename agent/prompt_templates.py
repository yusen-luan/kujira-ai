"""Prompts + response parsing for the propose/repair/synthesis/diagnosis/probe LLM calls.

Output contract for a normal axis propose call (enforced by parse_response, not by
--json-schema, since we want full raw file contents rather than JSON-escaped strings):

    REFLECTION: <1-3 sentences on the last iteration on this axis -- omitted on a
    first iteration>

    HYPOTHESIS: <1-3 sentences>
    EXPECTED_EFFECT: <1 sentence: which metric you expect to move more, and why>

    ```python:data.py
    <full new file content, only if changed>
    ```

    ```python:baseline.py
    <full new file content, only if changed>
    ```

Any file block whose name isn't in ALLOWED_FILES is dropped by the caller.

Axes: the hypothesis tree branches along a small fixed set of axes (see
AGENT.md "Next focus" / agent_notes/orchestrator.md), each with its own system
prompt (what it's allowed/encouraged to change) but the same ALLOWED_FILES and
output format. `feature` and `model` are the two axes that need LLM
creativity; hyperparameter search is deliberately NOT an axis here — it's
cheap enough to do with a plain sweep, no LLM call needed (see orchestrator.py
if/when that's added).

Reflection loop (added 2026-08-29): each axis propose call is shown its own
last iteration's hypothesis + stated expected effect + actual result, and is
asked to write a REFLECTION on it before proposing the next hypothesis --
turning the per-axis history from a numeric ledger into something the model
has to reconcile its own predictions against. EXPECTED_EFFECT gives that
reflection something concrete to check (which per-metric direction was
predicted vs what actually moved), not just "did primary go up."

Diagnosis + probe loop (added 2026-08-29): when an axis racks up consecutive
rejections/failures, orchestrator.py spends a reserved node on a DIAGNOSIS
call (this axis's recent hypotheses + reflections -> a root-cause category:
data_gap / modeling_ceiling / low_diversity / engineering). Only data_gap
triggers a follow-up PROBE call: the LLM writes a small read-only numpy
computation (probe.py, contract: run_probe(arrays, masks, user_feat,
video_feat, label) -> dict) answering the specific question the diagnosis
raised, executed by agent/eda_probe.py against the same pre-loaded objects
eda.py's own report uses -- no new filesystem-access surface, no rewriting of
the pinned eda_report.json. Results accumulate in
agent/runs/probe_findings.md and get pasted into every future propose prompt
alongside the EDA summary and literature context.
"""
import re
from pathlib import Path

ALLOWED_FILES = ('data.py', 'baseline.py')
PROBE_ALLOWED_FILES = ('probe.py',)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_ZOO_DIR = _REPO_ROOT / 'workspace' / 'models'
_MODEL_ZOO_FILES = ('fm.py', 'dcn.py', 'dfm.py', 'afi.py', 'xdfm.py')
_EDA_SUMMARY_PATH = _REPO_ROOT / 'agent' / 'runs' / 'eda_summary.md'
_LITERATURE_CONTEXT_PATH = _REPO_ROOT / 'agent' / 'runs' / 'literature_context.md'
_PROBE_FINDINGS_PATH = _REPO_ROOT / 'agent' / 'runs' / 'probe_findings.md'


def _load_literature_context():
    """Read lazily at prompt-build time, same reasoning as _load_eda_summary below:
    orchestrator.py generates this file (via rag.py) before the first propose call,
    but after prompt_templates is imported."""
    if _LITERATURE_CONTEXT_PATH.exists():
        text = _LITERATURE_CONTEXT_PATH.read_text(encoding='utf-8').strip()
        if text:
            return text
    return ('(no literature retrieved -- run `python agent/rag.py` first, or it runs '
            'automatically before the first propose call in orchestrator.py.)')


def _load_eda_summary():
    """Read at prompt-build time (not import time): orchestrator.py generates this
    file via eda.py before the first propose call in a run, but prompt_templates is
    imported earlier than that, so baking this into a module-level constant would
    freeze in whatever was on disk at import -- usually nothing yet."""
    if _EDA_SUMMARY_PATH.exists():
        text = _EDA_SUMMARY_PATH.read_text(encoding='utf-8').strip()
        if text:
            return text
    return ('(no EDA report found -- run `python agent/eda.py` first. Proceeding '
            'without data-grounded facts; treat any assumption about class balance, '
            'cardinality, or missing-value handling as unverified.)')


def _load_probe_findings():
    """Read lazily, same pattern as EDA summary / literature context: orchestrator.py
    appends to this file over the course of a run (agent/runs/probe_findings.md), so
    it must never be baked into a module-level constant."""
    if _PROBE_FINDINGS_PATH.exists():
        text = _PROBE_FINDINGS_PATH.read_text(encoding='utf-8').strip()
        if text:
            return text
    return '(no diagnostic probes have been run yet this project.)'


def _load_model_zoo_reference():
    """Inlines the torch model zoo source into the model-axis prompt.

    llm.py calls the LLM with --tools "" (pure text completion, no filesystem
    access) so it cannot read these files itself — anything it should be able
    to adapt has to be pasted into the prompt text."""
    parts = []
    for fname in _MODEL_ZOO_FILES:
        path = _MODEL_ZOO_DIR / fname
        if not path.exists():
            continue
        parts.append(f"--- workspace/models/{fname} ---\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


_CONTRACT = """Hard constraints you must preserve, because a fixed harness calls into these files \
directly and cannot be changed:
- data.py must keep a `load(data_dir)` function returning the same split dict shape, \
and an `encode(splits)` function returning `(enc, dim)` where `enc[name] = (X, y, users)`.
- baseline.py must keep a `run_model(splits, hparams: dict, seed=0, verbose=True)` \
function returning `{'valid': {...}, 'test': {...}}` (each a dict with GAUC, nDCG@5, \
primary) — this is the fixed entrypoint the harness calls, so it must exist with this \
exact name and signature no matter what else you change. The harness always passes the \
same generic hparams dict regardless of which axis you're working on, so don't assume it \
contains any particular keys — read what you need via `hparams.get(name, default)` with \
sensible defaults, and don't raise if a key you'd expect is missing or one you don't use \
is present.
- When `verbose=True`, print one line per epoch to stdout (epoch number, training loss, \
valid GAUC/nDCG@5/primary, and the epoch's wall time) — this is streamed live to the user's \
terminal so they can watch training progress, so keep it to one concise line per epoch, not more.
- Don't try to change evaluate.py's scoring semantics — you won't be shown that file and \
can't edit it anyway.
- Only propose changes to two files: data.py and baseline.py. Anything else you write is \
ignored.
- Never use a same-row post-impression outcome field as a raw input feature: play_time_ms, \
profile_stay_time, comment_stay_time, is_click, is_like, is_follow, is_comment, is_forward, \
is_hate, is_profile_enter. These are recorded concurrently with the row's own label and \
wouldn't be known yet at serving time in a real system — using them directly is a label leak, \
not a real feature. An *aggregated historical* version of one of these (e.g. a user's \
long_view rate over their own prior rows, computed so it never looks at the current row) is \
fine and is a legitimate feature-axis idea.
- If a retrieved literature note or a diagnostic probe finding below directly informed your \
hypothesis, name it (e.g. "per ESMM..." or "per the probe on node 7...") in the HYPOTHESIS \
line — this is tracked for the project's write-up. Don't force a citation where none applies; \
a hypothesis with no literature/probe basis is fine too."""

_CODE_BLOCKS_FORMAT = """```python:data.py
<full new file content, only if you are changing this file>
```

```python:baseline.py
<full new file content, only if you are changing this file>
```

Omit a file's fenced block entirely if you are not changing it. Output nothing else \
outside this format — no preamble, no explanation after the code."""

# Plain format (no reflection/expected-effect): used by synthesis, which merges two
# already-formed candidates rather than iterating on its own history.
_OUTPUT_FORMAT = f"""Output format, exactly:

HYPOTHESIS: <1-3 sentences: what you're changing and why you think it will help>

{_CODE_BLOCKS_FORMAT}"""

# Axis format: adds REFLECTION (on the axis's own last iteration, shown to the model in
# the user prompt) and EXPECTED_EFFECT (a checkable prediction, graded by the next
# iteration's REFLECTION). See module docstring for why.
_OUTPUT_FORMAT_AXIS = f"""Output format, exactly:

REFLECTION: <1-3 sentences reflecting on your last iteration on this axis (shown to you \
below, under "Your last iteration on this axis") -- did the outcome match what you \
expected, and what's the most likely explanation either way? Omit this line entirely if \
this is your first iteration on this axis.>

HYPOTHESIS: <1-3 sentences: what you're changing and why you think it will help>
EXPECTED_EFFECT: <1 sentence: which metric (GAUC vs nDCG@5) you expect to move more, and \
why -- this is checked against the actual result and referenced in your next iteration's \
REFLECTION, so commit to a real, falsifiable prediction rather than a hedge>

{_CODE_BLOCKS_FORMAT}"""


FEATURE_SYSTEM_PROMPT = f"""You are an ML engineering agent iterating on a recommendation-ranking \
baseline (KuaiRand-Pure). You are exploring the FEATURE AXIS this turn: feature engineering and \
preprocessing (new fields, encodings, negative sampling, normalization, interaction counts, etc.) \
in data.py. Don't change the model/training algorithm itself on this axis — that's a different \
axis's job. You may touch baseline.py only if strictly necessary to plug a new feature in (e.g. \
widening the FM's field count); don't restructure the model.

{_CONTRACT}
- numpy and the Python standard library only on this axis. No new pip dependencies, no torch, no \
sklearn, no internet access.

Your job each turn: propose exactly ONE focused hypothesis for improving the validation `primary` \
metric (mean of GAUC and nDCG@5), then rewrite the full contents of whichever of data.py / \
baseline.py that hypothesis requires. Keep it to one attributable change per turn (one feature, \
one encoding tweak) — not a grab-bag of unrelated changes, since we need to be able to tell what \
caused any metric change.

{_OUTPUT_FORMAT_AXIS}"""


MODEL_SYSTEM_PROMPT = f"""You are an ML engineering agent iterating on a recommendation-ranking \
baseline (KuaiRand-Pure). You are exploring the MODEL AXIS this turn: the model architecture and \
training procedure in baseline.py. Don't add new features to data.py on this axis (that's a \
different axis's job) — you may touch data.py only if your model genuinely needs a different \
input encoding (e.g. padding for a sequence model), not to add signal.

torch and torchfm are installed and available to import on this axis (they are NOT available on \
the feature axis — numpy-only there). Reference implementations of several architectures (FM, \
AutoInt/AFI, DCN, DeepFM, xDeepFM) are pasted below, adapted from a prior project of the user's. \
Prioritize adapting/importing these over writing a new architecture from scratch — they're already \
implemented and tested, so reuse is cheaper and less failure-prone than fresh generation. You may \
also propose a numpy-only architectural change (no new fields to FM's math, a different training \
scheme) if that's a better fit for the run-time budget than a torch model.

{_CONTRACT}
- No new pip dependencies beyond torch + torchfm (both already installed). No internet access, no \
downloading pretrained weights.
- Mind the wall-clock budget: this candidate must finish training within the run timeout on CPU \
(no GPU available) — keep epoch count and model size modest. This is not the place to reproduce a \
paper's full training regimen; a smaller/faster version of an architecture that actually finishes \
and beats the FM baseline is better than a bigger one that times out.

Your job each turn: propose exactly ONE focused hypothesis for improving the validation `primary` \
metric (mean of GAUC and nDCG@5) via the model/training procedure, then rewrite the full contents \
of whichever of data.py / baseline.py that hypothesis requires. Keep it to one attributable change \
per turn — not a grab-bag of unrelated changes.

{_OUTPUT_FORMAT_AXIS}

Reference model zoo (adapt from these where useful — field_dims/embed_dim-style constructors, \
takes a LongTensor of shape (batch, num_fields)):

{_load_model_zoo_reference()}"""


SYNTHESIS_SYSTEM_PROMPT = f"""You are an ML engineering agent. Two independent improvements were \
made to the same starting baseline along different axes: one improved feature engineering \
(data.py), the other improved the model/training procedure (baseline.py, possibly using PyTorch). \
Your job is to combine both into a single working candidate that keeps both improvements — not to \
invent a new idea.

{_CONTRACT}
- torch/torchfm are available if the model-axis candidate used them.

{_OUTPUT_FORMAT}"""


DIAGNOSIS_SYSTEM_PROMPT = """You are an ML engineering agent diagnosing why a series of hypotheses \
on one axis of a recommendation-ranking pipeline (KuaiRand-Pure) failed to improve the validation \
`primary` metric. You are given the recent hypotheses tried on this axis, what each expected to \
happen, what actually happened, and the reflection written at the time (if any). Your job is NOT to \
propose a new hypothesis — it's to classify the most likely root cause, so the orchestrator can \
decide what kind of help to give next. Getting this classification right matters: triggering a data \
probe when the real problem is something else wastes a turn, and vice versa.

Output format, exactly:

CATEGORY: <one of: data_gap, modeling_ceiling, low_diversity, engineering>
RATIONALE: <1-3 sentences justifying the category, citing the specific pattern across the recent \
attempts>
PROBE_QUESTION: <required only if CATEGORY is data_gap — a single, specific, answerable question \
about the raw data that a short numpy computation could resolve (e.g. "what is the correlation \
between a user's rolling historical long_view rate and their current-row long_view label, and does \
it differ by tab?"), one you believe nothing computed so far (EDA report, prior probes) has already \
answered, and which -- if answered -- would tell you whether the mechanism these failed hypotheses \
relied on actually exists in the data. Omit this line entirely for any other category.>

Category definitions:
- data_gap: the failed hypotheses all assumed some data property/mechanism that nothing computed so \
far (the EDA report, prior probes) actually verified — the real problem is that nobody has looked at \
the right thing yet, not that the idea itself was bad. This is usually the right call when the \
reflections describe *surprise* at the outcome (the mechanism should have worked but didn't, for \
unclear reasons) rather than a shrug.
- modeling_ceiling: the data property the hypotheses relied on IS already confirmed by EDA/prior \
probes, but this axis appears to be near its capacity (feature axis genuinely out of cheap signal to \
add, or model axis's architectures aren't beating a well-tuned FM within the time budget) — more data \
analysis won't help here; a different kind of change might.
- low_diversity: the failed hypotheses are variants of the same underlying mechanism (e.g. three ways \
of bucketing the same popularity signal) — there's no data mystery, the fix is trying a genuinely \
different mechanism next, not more analysis.
- engineering: failures were driven by bugs, timeouts, or repair-loop churn rather than the metric \
telling you anything about the hypothesis's merit.

Output nothing else outside this format."""


PROBE_SYSTEM_PROMPT = """You are an ML engineering agent answering one specific, narrow question \
about a recommendation-ranking dataset (KuaiRand-Pure) by writing a short read-only numpy \
computation. This is NOT a model-improvement step — you're not proposing a feature or model change, \
just computing a number that resolves a specific uncertainty raised during diagnosis.

You do not have file access. You are given, as plain function arguments, the exact same pre-loaded \
objects agent/eda.py's own analysis uses internally:
- `arrays`: dict[str, np.ndarray] — one int64 array per raw log column (user_id, video_id, date, \
hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, \
play_time_ms, duration_ms, profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab), \
each aligned row-for-row across both log files concatenated.
- `masks`: dict['train' | 'valid' | 'test' -> boolean np.ndarray] selecting each split's rows (same \
length as every array in `arrays`).
- `user_feat` / `video_feat`: dict[id_str -> dict[column_name -> string_value]], parsed straight from \
user_features_pure.csv / video_features_basic_pure.csv — every value is a raw string, cast as needed \
(and note some numeric-looking columns carry sentinel values, e.g. -124 for a missing flag — see the \
EDA summary below).
- `label`: the label column name (str) — currently 'long_view'. Index into `arrays[label]`.

Write exactly one function, named exactly `run_probe`, in a file named exactly `probe.py`:

```python:probe.py
def run_probe(arrays, masks, user_feat, video_feat, label):
    ...
    return {...}  # must be a JSON-serializable dict of your findings
```

Hard constraints:
- numpy and the Python standard library only. No new pip dependencies, no torch/sklearn, no file \
I/O, no network access — you already have everything you need as arguments.
- Read-only: don't mutate `arrays`/`masks`/`user_feat`/`video_feat` in place.
- Never treat a same-row post-impression outcome field as if it were known ahead of the label: \
play_time_ms, profile_stay_time, comment_stay_time, and every feedback signal other than the label \
itself are recorded concurrently with the row's own outcome (same rule as the main feature/model \
axes) — using one directly to "explain" the label is circular, not a finding.
- Keep it fast: this must finish in well under a minute on CPU. Compute exactly what answers the \
question — don't build a general-purpose analysis tool.
- Return a small number of named statistics (floats/ints/short strings/short lists), not raw arrays \
or large tables — this result gets pasted verbatim into future prompts, so keep it compact and \
directly interpretable.

Output format, exactly:

HYPOTHESIS: <restate the question you're answering, in one sentence>

```python:probe.py
<full file content>
```

Output nothing else outside this format."""


AXES = {
    'feature': {'label': 'feature engineering', 'system_prompt': FEATURE_SYSTEM_PROMPT},
    'model': {'label': 'model / architecture', 'system_prompt': MODEL_SYSTEM_PROMPT},
}


def _format_history(history):
    if not history:
        return "(no iterations yet)"
    lines = []
    for h in history:
        axis_tag = f"[{h['axis']}] " if h.get('axis') else ""
        status = h['status']
        if status == 'accepted':
            lines.append(f"iter {h['iter']}: {axis_tag}ACCEPTED \"{h['hypothesis']}\" "
                         f"-> primary {h['primary']:.4f} (was {h['prev_best']:.4f})")
        elif status == 'rejected':
            lines.append(f"iter {h['iter']}: {axis_tag}rejected (no improvement) \"{h['hypothesis']}\" "
                         f"-> primary {h['primary']:.4f} (best stays {h['prev_best']:.4f})")
        elif status == 'diagnosed':
            lines.append(f"iter {h['iter']}: {axis_tag}diagnosis -> category={h.get('category')}")
        elif status == 'answered':
            lines.append(f"iter {h['iter']}: {axis_tag}probe answered (see diagnostic probe "
                         f"findings below): \"{h.get('question', '')}\"")
        else:  # failed
            lines.append(f"iter {h['iter']}: {axis_tag}FAILED after retries \"{h.get('hypothesis')}\" "
                         f"-> {h.get('error_summary')}")
    return "\n".join(lines)


def _last_axis_entry(history, axis):
    """Most recent history entry that is itself a hypothesis attempt on `axis` (skips
    that axis's own diagnosis/probe entries, which are tagged '<axis>_diagnosis' /
    '<axis>_probe', not `axis` — see run_diagnosis/run_probe in orchestrator.py)."""
    for h in reversed(history):
        if h.get('axis') == axis:
            return h
    return None


def _format_reflection_block(axis, history):
    last = _last_axis_entry(history, axis)
    if last is None:
        return "(this is your first iteration on this axis — nothing to reflect on yet.)"
    lines = [f"Hypothesis: {last.get('hypothesis')}"]
    if last.get('expected_effect'):
        lines.append(f"Expected effect (stated at the time): {last['expected_effect']}")
    if last['status'] == 'failed':
        lines.append(f"Result: FAILED after retries -- {last.get('error_summary')}")
    else:
        lines.append(f"Result: {last['status']} -- valid primary {last['primary']:.4f} "
                     f"(was {last['prev_best']:.4f})")
        v = last.get('metrics', {}).get('valid', {})
        prev_v = last.get('prev_metrics', {}).get('valid', {})
        if v and prev_v:
            # Show the actual before->after delta per metric, since EXPECTED_EFFECT is a
            # claim about which metric moved MORE -- an absolute number alone can't be
            # checked against that, only a delta can.
            lines.append(f"Actual per-metric change: GAUC {prev_v.get('GAUC'):.4f} -> "
                         f"{v.get('GAUC'):.4f} ({v.get('GAUC') - prev_v.get('GAUC'):+.4f}), "
                         f"nDCG@5 {prev_v.get('nDCG@5'):.4f} -> {v.get('nDCG@5'):.4f} "
                         f"({v.get('nDCG@5') - prev_v.get('nDCG@5'):+.4f})")
        elif v:
            lines.append(f"Actual per-metric valid result: GAUC={v.get('GAUC'):.4f}, "
                         f"nDCG@5={v.get('nDCG@5'):.4f}")
    return "\n".join(lines)


def build_propose_prompt(axis, best_code, history, best_primary):
    hist_txt = _format_history(history)
    axis_label = AXES[axis]['label']
    eda_txt = _load_eda_summary()
    lit_txt = _load_literature_context()
    probe_txt = _load_probe_findings()
    reflect_txt = _format_reflection_block(axis, history)
    return f"""Your last iteration on this axis (reflect on this before proposing your next \
hypothesis -- did the result match what you expected, and what's the most likely explanation \
either way?):
{reflect_txt}

Data facts from EDA (computed once, directly from the real CSVs -- treat as ground \
truth, not something to re-derive from common sense):
{eda_txt}

Findings from targeted diagnostic probes run so far on this project (if any -- these answer \
specific questions raised when an axis got stuck, and are also ground truth):
{probe_txt}

Relevant published methods (retrieved from a small curated corpus, selected based on the EDA \
findings above -- not an exhaustive literature review, just what's most likely relevant here):
{lit_txt}

Current best validation primary metric on the {axis_label} axis: {best_primary:.4f}

History of past iterations (all axes, for context — avoid repeating what's already been tried):
{hist_txt}

Current data.py (this axis's branch):
```python
{best_code['data.py']}
```

Current baseline.py (this axis's branch):
```python
{best_code['baseline.py']}
```

Propose your next hypothesis and code change now, following the output format from the \
system prompt."""


def build_repair_prompt(attempt_code, hypothesis, error_text):
    files_txt = "\n\n".join(
        f"Current {fname}:\n```python\n{content}\n```"
        for fname, content in attempt_code.items()
    )
    return f"""Your previous attempt for this hypothesis failed to run.

HYPOTHESIS: {hypothesis}

Error output:
```
{error_text[-4000:]}
```

{files_txt}

Fix the bug. Keep the same hypothesis text. Output ONLY the HYPOTHESIS: line + fenced code \
block(s) in the exact format from the system prompt — no REFLECTION or EXPECTED_EFFECT needed \
for a repair."""


def build_synthesis_prompt(feature_code, feature_hypothesis, model_code, model_hypothesis, history):
    hist_txt = _format_history(history)
    eda_txt = _load_eda_summary()
    return f"""Data facts from EDA (computed once, directly from the real CSVs -- treat as ground \
truth, not something to re-derive from common sense):
{eda_txt}

History of past iterations:
{hist_txt}

Feature-axis candidate — HYPOTHESIS: {feature_hypothesis}

Feature-axis data.py:
```python
{feature_code['data.py']}
```

Feature-axis baseline.py:
```python
{feature_code['baseline.py']}
```

Model-axis candidate — HYPOTHESIS: {model_hypothesis}

Model-axis data.py:
```python
{model_code['data.py']}
```

Model-axis baseline.py:
```python
{model_code['baseline.py']}
```

Combine both into one candidate now, following the output format from the system prompt."""


def build_diagnosis_prompt(axis, history):
    axis_label = AXES[axis]['label']
    recent = [h for h in history if h.get('axis') == axis][-4:]
    lines = []
    for h in recent:
        lines.append(f"iter {h['iter']}: {h['status']} — HYPOTHESIS: {h.get('hypothesis')}")
        if h.get('expected_effect'):
            lines.append(f"  expected effect: {h['expected_effect']}")
        metrics, prev_metrics = h.get('metrics'), h.get('prev_metrics')
        if metrics and prev_metrics:
            v, pv = metrics.get('valid', {}), prev_metrics.get('valid', {})
            lines.append(f"  actual change: GAUC {pv.get('GAUC')}->{v.get('GAUC')}, "
                         f"nDCG@5 {pv.get('nDCG@5')}->{v.get('nDCG@5')}, "
                         f"primary {pv.get('primary')}->{v.get('primary')}")
        elif metrics:
            v = metrics.get('valid', {})
            lines.append(f"  actual valid result: GAUC={v.get('GAUC')}, nDCG@5={v.get('nDCG@5')}, "
                         f"primary={v.get('primary')} (prev best {h.get('prev_best')})")
        if h.get('status') == 'failed':
            lines.append(f"  error: {h.get('error_summary')}")
        if h.get('reflection'):
            lines.append(f"  reflection written at the time: {h['reflection']}")
    recent_txt = "\n".join(lines) if lines else "(no hypothesis attempts recorded on this axis yet)"
    eda_txt = _load_eda_summary()
    probe_txt = _load_probe_findings()
    return f"""Axis under diagnosis: {axis_label}

Data facts from EDA:
{eda_txt}

Findings from prior diagnostic probes on this project (if any):
{probe_txt}

Recent attempts on this axis, most recent last:
{recent_txt}

Diagnose the root cause now, following the output format from the system prompt."""


def build_probe_prompt(question, axis):
    axis_label = AXES[axis]['label']
    eda_txt = _load_eda_summary()
    probe_txt = _load_probe_findings()
    return f"""A diagnosis on the {axis_label} axis raised this question, believed unanswered by \
the EDA report or any prior probe so far:

{question}

Data facts from EDA (for context — column names, splits, known data-quality issues, etc.):
{eda_txt}

Findings from prior diagnostic probes (avoid recomputing something already answered here):
{probe_txt}

Write run_probe now, following the output format from the system prompt."""


_FENCE_RE = re.compile(r"```python:([^\n`]+)\n(.*?)```", re.DOTALL)
_REFLECTION_RE = re.compile(r"REFLECTION:\s*(.*?)(?:\n\s*HYPOTHESIS:|\Z)", re.DOTALL)
_HYPOTHESIS_RE = re.compile(r"HYPOTHESIS:\s*(.*?)(?:\n\s*EXPECTED_EFFECT:|\n```|\Z)", re.DOTALL)
_EXPECTED_EFFECT_RE = re.compile(r"EXPECTED_EFFECT:\s*(.*?)(?:\n```|\Z)", re.DOTALL)


def parse_response(text):
    """Returns a dict: {'reflection': str, 'hypothesis': str, 'expected_effect': str,
    'files': dict[filename -> content]}. reflection/expected_effect are '' when the
    model omitted them — expected for a repair call, a synthesis call, or a first-ever
    iteration on an axis, all of which legitimately have nothing to put there."""
    files = {}
    for m in _FENCE_RE.finditer(text):
        fname = m.group(1).strip()
        content = m.group(2)
        if content.endswith("\n"):
            content = content[:-1]
        files[fname] = content
    reflection_match = _REFLECTION_RE.search(text)
    reflection = reflection_match.group(1).strip() if reflection_match else ""
    hyp_match = _HYPOTHESIS_RE.search(text)
    hypothesis = hyp_match.group(1).strip() if hyp_match else ""
    ee_match = _EXPECTED_EFFECT_RE.search(text)
    expected_effect = ee_match.group(1).strip() if ee_match else ""
    return {'reflection': reflection, 'hypothesis': hypothesis,
            'expected_effect': expected_effect, 'files': files}


_CATEGORY_RE = re.compile(r"CATEGORY:\s*(\w+)")
_RATIONALE_RE = re.compile(r"RATIONALE:\s*(.*?)(?:\n\s*PROBE_QUESTION:|\Z)", re.DOTALL)
_PROBE_QUESTION_RE = re.compile(r"PROBE_QUESTION:\s*(.*)", re.DOTALL)

VALID_CATEGORIES = {'data_gap', 'modeling_ceiling', 'low_diversity', 'engineering'}


def parse_diagnosis_response(text):
    """Returns {'category': str, 'rationale': str, 'probe_question': str}. Falls back
    to category='engineering' (the no-op category — just try a different hypothesis
    next, don't spend a probe) if CATEGORY is missing or not one of VALID_CATEGORIES,
    so a malformed diagnosis response fails safe rather than spending probe budget on
    a category it never actually named."""
    cat_match = _CATEGORY_RE.search(text)
    category = cat_match.group(1).strip() if cat_match else 'engineering'
    if category not in VALID_CATEGORIES:
        category = 'engineering'
    rat_match = _RATIONALE_RE.search(text)
    rationale = rat_match.group(1).strip() if rat_match else ''
    probe_question = ''
    if category == 'data_gap':
        pq_match = _PROBE_QUESTION_RE.search(text)
        probe_question = pq_match.group(1).strip() if pq_match else ''
    return {'category': category, 'rationale': rationale, 'probe_question': probe_question}
