"""Prompts + response parsing for the propose/repair/synthesis LLM calls.

Output contract the LLM must follow (enforced by parse_response, not by
--json-schema, since we want full raw file contents rather than
JSON-escaped strings):

    HYPOTHESIS: <1-3 sentences>

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
"""
import re
from pathlib import Path

ALLOWED_FILES = ('data.py', 'baseline.py')

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_ZOO_DIR = _REPO_ROOT / 'workspace' / 'models'
_MODEL_ZOO_FILES = ('fm.py', 'dcn.py', 'dfm.py', 'afi.py', 'xdfm.py')
_EDA_SUMMARY_PATH = _REPO_ROOT / 'agent' / 'runs' / 'eda_summary.md'


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
fine and is a legitimate feature-axis idea."""

_OUTPUT_FORMAT = """Output format, exactly:

HYPOTHESIS: <1-3 sentences: what you're changing and why you think it will help>

```python:data.py
<full new file content, only if you are changing this file>
```

```python:baseline.py
<full new file content, only if you are changing this file>
```

Omit a file's fenced block entirely if you are not changing it. Output nothing else \
outside this format — no preamble, no explanation after the code."""


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

{_OUTPUT_FORMAT}"""


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

{_OUTPUT_FORMAT}

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
        else:  # failed
            lines.append(f"iter {h['iter']}: {axis_tag}FAILED after retries \"{h['hypothesis']}\" "
                         f"-> {h['error_summary']}")
    return "\n".join(lines)


def build_propose_prompt(axis, best_code, history, best_primary):
    hist_txt = _format_history(history)
    axis_label = AXES[axis]['label']
    eda_txt = _load_eda_summary()
    return f"""Data facts from EDA (computed once, directly from the real CSVs -- treat as ground \
truth, not something to re-derive from common sense):
{eda_txt}

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

Fix the bug. Keep the same hypothesis text. Output the corrected full file(s) using the \
exact format from the system prompt (HYPOTHESIS: line + fenced code block(s))."""


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


_FENCE_RE = re.compile(r"```python:([^\n`]+)\n(.*?)```", re.DOTALL)
_HYPOTHESIS_RE = re.compile(r"HYPOTHESIS:\s*(.*?)(?:\n```|\Z)", re.DOTALL)


def parse_response(text):
    """Returns (hypothesis: str, files: dict[filename -> content])."""
    files = {}
    for m in _FENCE_RE.finditer(text):
        fname = m.group(1).strip()
        content = m.group(2)
        if content.endswith("\n"):
            content = content[:-1]
        files[fname] = content
    hyp_match = _HYPOTHESIS_RE.search(text)
    hypothesis = hyp_match.group(1).strip() if hyp_match else ""
    return hypothesis, files
