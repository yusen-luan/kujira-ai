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

Two possible actions per turn (mutually exclusive, decided by the LLM itself,
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

Plateau escalation (replaces v1's --stuck_after/diagnosis_budget machinery):
orchestrator.py tracks a plain consecutive-iterations-without-a-real-
improvement streak (using the *official* epsilon=0.002, not the loose
accept/reject noise floor) and once it crosses --escalate_after, the propose
prompt gets an explicit "you're plateauing, don't propose a variant of what
you just tried" instruction, naming the recent hypotheses so the LLM has
something concrete to diverge from. Probes consume a normal --iterations
slot now (no separate reserved budget) -- simpler, and self-limiting since
spending the whole run on probes instead of code changes doesn't move the
metric either.
"""
import re
from pathlib import Path

ALLOWED_FILES = ('data.py', 'baseline.py')
PROBE_ALLOWED_FILES = ('probe.py',)
MAX_SWEEP_VALUES = 4

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


_CONTRACT = """Hard constraints you must preserve, because a fixed harness calls into these files \
directly and cannot be changed:
- data.py must keep a `load(data_dir)` function returning the same split dict shape, \
and an `encode(splits)` function returning `(enc, dim)` where `enc[name] = (X, y, users)`.
- baseline.py must keep a `run_model(splits, hparams: dict, seed=0, verbose=True)` \
function returning `{'valid': {...}, 'test': {...}}` (each a dict with GAUC, nDCG@5, \
primary) — this is the fixed entrypoint the harness calls, so it must exist with this \
exact name and signature no matter what else you change. The harness always passes the \
same generic hparams dict regardless of what kind of change you're making, so don't assume \
it contains any particular keys — read what you need via `hparams.get(name, default)` with \
sensible defaults, and don't raise if a key you'd expect is missing or one you don't use \
is present.
- numpy and the Python standard library only in data.py. torch and torchfm are additionally \
allowed in baseline.py (not in data.py) if you're proposing a model/architecture change — both \
are installed. No other new pip dependencies, no internet access.
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
- One attributable change per turn: primarily touch ONE of data.py or baseline.py. You may touch \
the other file too only if strictly necessary to integrate that one change (e.g. widening a field \
count in data.py may require no baseline.py change at all, since encode() already reports `dim` \
generically) — don't make two independent, unrelated changes in the same turn, since we need to \
be able to tell what caused any metric change.
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

_OUTPUT_FORMAT = """Output format, exactly one of the following two:

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

Output nothing else outside one of these two formats — no preamble, no explanation after the code, \
and never both HYPOTHESIS and PROBE_QUESTION in the same response."""


SYSTEM_PROMPT = f"""You are an ML engineering agent iterating on a recommendation-ranking baseline \
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

{_CONTRACT}

{_PROBE_CONTRACT}

Your job each turn: propose exactly ONE focused hypothesis for improving the validation `primary` \
metric (mean of GAUC and nDCG@5) — via a feature, the model, or training/loss — and rewrite the \
full contents of whichever single file that requires; or, if you lack the information to do that \
well, write a probe instead. Keep it to one attributable change per turn, not a grab-bag of \
unrelated changes, since we need to be able to tell what caused any metric change.

{_OUTPUT_FORMAT}

Reference model zoo (only relevant if you're proposing a model/architecture change — \
field_dims/embed_dim-style constructors, takes a LongTensor of shape (batch, num_fields)):

{_load_model_zoo_reference()}"""


def _format_sweep(h):
    sweep = h.get('sweep')
    if not sweep:
        return ""
    parts = []
    for t in sweep['trials']:
        mark = '*' if sweep.get('best_value') == t['value'] and t['ok'] else ''
        parts.append(f"{t['value']}→{t['primary']:.4f}{mark}" if t['ok'] else f"{t['value']}→FAILED")
    return f" [swept {sweep['param']}: {', '.join(parts)}]"


def _format_history(history):
    if not history:
        return "(no iterations yet)"
    lines = []
    for h in history:
        status = h['status']
        if status == 'accepted':
            lines.append(f"iter {h['iter']}: ACCEPTED \"{h['hypothesis']}\"{_format_sweep(h)} "
                         f"-> primary {h['primary']:.4f} (was {h['prev_best']:.4f})")
        elif status == 'rejected':
            lines.append(f"iter {h['iter']}: rejected (no improvement) \"{h['hypothesis']}\"{_format_sweep(h)} "
                         f"-> primary {h['primary']:.4f} (best stays {h['prev_best']:.4f})")
        elif status == 'answered':
            lines.append(f"iter {h['iter']}: PROBE \"{h.get('question', '')}\" "
                         f"-> see diagnostic probe findings below")
        elif status == 'failed' and h.get('question') is not None:
            lines.append(f"iter {h['iter']}: probe FAILED \"{h.get('question', '')}\"")
        else:  # failed hypothesis turn
            lines.append(f"iter {h['iter']}: FAILED after retries \"{h.get('hypothesis')}\" "
                         f"-> {h.get('error_summary')}")
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
        if h.get('status') in ('accepted', 'rejected', 'failed', 'answered'):
            last = h
            break
    if last is None:
        return "(this is your first iteration — nothing to reflect on yet.)"
    if last['status'] == 'answered':
        return (f"Your last iteration ran a diagnostic probe rather than proposing a change: "
                f"\"{last.get('question')}\" — see the findings below for the answer.")
    lines = [f"Hypothesis: {last.get('hypothesis')}{_format_sweep(last)}"]
    if last.get('expected_effect'):
        lines.append(f"Expected effect (stated at the time): {last['expected_effect']}")
    if last['status'] == 'failed':
        lines.append(f"Result: FAILED after retries -- {last.get('error_summary')}")
        return "\n".join(lines)
    lines.append(f"Result: {last['status']} -- valid primary {last['primary']:.4f} "
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


def _format_escalation(streak, escalate_after, eps, history):
    if streak < escalate_after:
        return ""
    recent = [h for h in history if h.get('status') in ('accepted', 'rejected')][-streak:]
    recent_txt = "; ".join(f'"{(h.get("hypothesis") or "")[:90]}"' for h in recent) or "(none)"
    return (f"\n⚠ PLATEAU WARNING: your last {streak} iterations each improved validation primary "
            f"by {eps} or less — you appear to be stuck reapplying variants of the same underlying "
            f"mechanism. Recent hypotheses: {recent_txt}. This turn, do NOT propose another small "
            f"variant of the same mechanism as those (e.g. another 'smoothed historical rate for a "
            f"different entity' if that's the pattern above). Either propose something structurally "
            f"different — a different kind of feature transformation, a different model architecture, "
            f"a different training objective/loss, a different negative-sampling or regularization "
            f"scheme — or write a probe to investigate a specific uncertainty about the data before "
            f"proposing again.\n")


def build_propose_prompt(best_code, history, best_primary, streak, escalate_after, converge_eps):
    hist_txt = _format_history(history)
    eda_txt = _load_eda_summary()
    lit_txt = _load_literature_context()
    probe_txt = _load_probe_findings()
    reflect_txt = _format_reflection_block(history)
    escalate_txt = _format_escalation(streak, escalate_after, converge_eps, history)
    return f"""Your last iteration (reflect on this before deciding your next move):
{reflect_txt}
{escalate_txt}
Data facts from EDA (computed once, directly from the real CSVs -- treat as ground truth, not \
something to re-derive from common sense):
{eda_txt}

Findings from targeted diagnostic probes run so far (if any -- these answer specific questions \
raised along the way, and are also ground truth):
{probe_txt}

Relevant published methods (retrieved from a small curated corpus, selected based on the EDA \
findings above -- not an exhaustive literature review, just what's most likely relevant here):
{lit_txt}

Current best validation primary metric: {best_primary:.4f}

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
_REFLECTION_RE = re.compile(r"REFLECTION:\s*(.*?)(?:\n\s*(?:HYPOTHESIS|PROBE_QUESTION):|\Z)", re.DOTALL)
_HYPOTHESIS_RE = re.compile(r"HYPOTHESIS:\s*(.*?)(?:\n\s*EXPECTED_EFFECT:|\n```|\Z)", re.DOTALL)
_EXPECTED_EFFECT_RE = re.compile(r"EXPECTED_EFFECT:\s*(.*?)(?:\n\s*SWEEP_PARAM:|\n```|\Z)", re.DOTALL)
_PROBE_QUESTION_RE = re.compile(r"PROBE_QUESTION:\s*(.*?)(?:\n```|\Z)", re.DOTALL)
_SWEEP_PARAM_RE = re.compile(r"SWEEP_PARAM:\s*(.*?)(?:\n\s*SWEEP_VALUES:|\n```|\Z)", re.DOTALL)
_SWEEP_VALUES_RE = re.compile(r"SWEEP_VALUES:\s*(.*?)(?:\n```|\Z)", re.DOTALL)


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
    'sweep_param', 'sweep_values', 'mode', 'files'}. `mode` is 'probe' if a probe.py file was
    produced and neither data.py nor baseline.py was (code-based detection, not text-label-based,
    so it's robust to the model accidentally leaving stray HYPOTHESIS/PROBE_QUESTION text around);
    otherwise 'hypothesis'. `sweep_param` is '' and `sweep_values` is [] unless both SWEEP_PARAM
    and at least one valid numeric SWEEP_VALUES entry were present -- a sweep is only ever
    opt-in, so a normal hypothesis turn (the vast majority) is unaffected by this parsing.
    Every field defaults to '' / {} when absent — expected for a repair call (no REFLECTION/
    HYPOTHESIS/EXPECTED_EFFECT/PROBE_QUESTION needed there) or a first-ever iteration."""
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
    pq_match = _PROBE_QUESTION_RE.search(text)
    probe_question = pq_match.group(1).strip() if pq_match else ""

    has_pipeline_files = any(f in ALLOWED_FILES for f in files)
    has_probe_file = 'probe.py' in files
    mode = 'probe' if has_probe_file and not has_pipeline_files else 'hypothesis'

    return {'reflection': reflection, 'hypothesis': hypothesis, 'expected_effect': expected_effect,
            'probe_question': probe_question, 'sweep_param': sweep_param, 'sweep_values': sweep_values,
            'mode': mode, 'files': files}
