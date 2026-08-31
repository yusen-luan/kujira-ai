"""Agentic exploratory-data-analysis loop -- v4 roadmap Phase 2.

A role structurally similar to repo_explore.py/web_research.py (own SYSTEM_PROMPT, own
prompt-builder, own response parser, own entry point) except its "tool" is writing a
probe.py run through the exact same fixed eda_probe.py harness the existing mid-run
PROBE_QUESTION mechanism already uses (via probe_runner.py, extracted so this module and
orchestrator.py's own PROBE_QUESTION path share one implementation instead of two).

Unlike that existing mid-run probe mode (one pre-posed question, answered once, only
ever iterating on failure-repair), this is a bounded, genuinely exploratory loop: each
turn the model sees the deterministic EDA report plus every probe run so far (across
every invocation of this module, not just the current one) and decides whether to ask
another question or FINALIZE. It is invoked from two places:

  - agent/eda.py::run() -- mandatory bootstrap pass, once per project, whenever no
    eda_report.json/eda_summary.md exist yet (focus=None: nothing to focus on yet).
  - agent/orchestrator.py::run_iteration() -- an optional, budgeted, agent-CHOSEN mid-run
    round (mode == 'eda_round'), triggered when the propose LLM itself decides recent
    lack of progress looks like a data-understanding gap rather than a modeling gap
    (focus=the LLM's stated question/reason).

Either way, findings are durably appended to agent/runs/eda_probes/log.json (never
overwritten) and persist into every future propose prompt by feeding
accumulated_findings_text() into eda.summarize_with_llm()'s extra_context, which
regenerates eda_summary.md -- prompt_templates.py's existing _load_eda_summary() needs
no changes, it already pastes that one file's contents verbatim into every propose turn.

Never touches the CSVs itself: every probe executes in a subprocess via
probe_runner.run_probe_and_repair(), receiving only the pre-loaded arrays/masks/
user_feat/video_feat/label eda_probe.py's harness already loads -- same read-only
guarantee as the existing mid-run probe mechanism, described in eda_probe.py's own
docstring.
"""
import json
import re
from pathlib import Path

import eda  # safe: eda.py only imports this module lazily, inside eda.run(), never at
            # its own top level -- see eda.py's run() for the load-order note
import llm
import probe_runner
import prompt_templates as pt

MAX_CONSECUTIVE_FAILURES = 2  # stop early rather than burn the whole turn budget on a
                               # model that keeps writing broken probes


SYSTEM_PROMPT = """You are doing open-ended exploratory data analysis on a \
recommendation dataset (KuaiRand-Pure, an engagement-prediction ranking task) to inform \
an ML-engineering agent's next move. A separate, deterministic pass already computed \
label rates, cardinality, cold-start overlap, popularity skew, distribution shift, and \
known data-quality flags (given to you below as JSON) -- your job is to ask NEW, \
specific, numerically-answerable questions that pass didn't cover, if any genuinely \
useful ones occur to you. This is not one pre-posed diagnostic question (a separate, \
narrower mechanism used mid-run handles that) -- you decide for yourself, turn by turn, \
whether there's anything else worth checking before finalizing.

You have no file access, no Bash, no network. Each turn you are given, as plain \
in-memory function arguments (never as a path or URL you could otherwise reach for), \
exactly what the deterministic pass itself uses internally:
- `arrays`: dict[str, np.ndarray] -- one int64 array per raw log column: user_id, \
video_id, date, hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, \
is_hate, long_view, play_time_ms, duration_ms, profile_stay_time, comment_stay_time, \
is_profile_enter, is_rand, tab.
- `masks`: dict['train'|'valid'|'test' -> boolean np.ndarray], same row alignment as \
every array above.
- `user_feat` / `video_feat`: dict[id_str -> dict[column_name -> string_value]], the \
side-info files.
- `label`: the label column name (str) the pipeline currently predicts.

Each turn, output EXACTLY ONE of:

1) Another probe -- only if you have a genuinely new, specific, answerable question \
that neither the deterministic report nor any prior turn shown to you already answered. \
A same-row post-impression column (play_time_ms, profile_stay_time, comment_stay_time, \
or any is_click/is_like/is_follow/is_comment/is_forward/is_hate/is_profile_enter other \
than the label itself) must never be used as if it were a legitimate input feature in a \
probe that's checking predictive power -- that would be leaking the label into its own \
evidence.

REFLECTION: <1-2 sentences: what you learned from your last probe (if any this session), \
and why this next question is worth the cost of running it>
PROBE: <the specific question, one sentence>
```python:probe.py
def run_probe(arrays, masks, user_feat, video_feat, label):
    ...
    return {...}  # a small, JSON-serializable dict of named statistics
```
Constraints on probe.py: numpy/stdlib only, no file I/O, no network, read-only (never \
mutate the arguments), fast (well under a minute on CPU), return only what's needed to \
answer the stated question.

2) FINALIZE -- once you have enough to hand off, including immediately on your first \
turn if the deterministic report and prior findings below already cover what you'd \
otherwise ask:

REFLECTION: <1-2 sentences on why you're stopping here>
FINALIZE"""


def _schema_text():
    return (f"Raw log columns: {', '.join(eda.LOG_COLS)}\n"
            f"Feedback/outcome columns: {', '.join(eda.FEEDBACK_COLS)}\n"
            f"Leakage columns (never usable as same-row input features, current label is "
            f"'{eda.LABEL}'): {', '.join(eda.LEAKAGE_COLS)}")


def _log_path(out_dir):
    return Path(out_dir) / 'eda_probes' / 'log.json'


def _load_log(out_dir):
    p = _log_path(out_dir)
    if not p.exists():
        return {'turns': []}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {'turns': []}


def _next_round(out_dir):
    log = _load_log(out_dir)
    if not log['turns']:
        return 0
    return max(t.get('round', 0) for t in log['turns']) + 1


def accumulated_findings_text(out_dir):
    """Markdown block of every successfully-answered probe ever run by this module
    (across every round, bootstrap included), for eda.summarize_with_llm()'s
    extra_context -- this is how a finding persists into every future propose prompt,
    since eda_summary.md is regenerated from this each time a round completes."""
    log = _load_log(out_dir)
    successes = [t for t in log['turns'] if t.get('mode') == 'probe' and 'result' in t]
    if not successes:
        return ''
    return "\n\n".join(
        f"### Round {t.get('round', 0)}, probe: {t['probe_question']}\n"
        f"```json\n{json.dumps(t['result'], indent=2)}\n```"
        for t in successes)


_PROBE_RE = re.compile(r"PROBE:\s*(.*?)(?:\n```|\Z)", re.DOTALL)
_REFLECTION_RE = re.compile(r"REFLECTION:\s*(.*?)(?:\nPROBE:|\nFINALIZE|\Z)", re.DOTALL)


def _parse_turn(text):
    """Returns {'mode': 'probe'|'finalize', 'reflection', 'probe_question', 'files'}.
    Defaults to 'finalize' on anything unparseable -- a malformed response must never
    spin the loop indefinitely; treat it the same as the model choosing to stop."""
    reflection_match = _REFLECTION_RE.search(text)
    reflection = reflection_match.group(1).strip() if reflection_match else ''
    files = pt.parse_fenced_files(text)
    if 'probe.py' in files:
        pq_match = _PROBE_RE.search(text)
        return {'mode': 'probe', 'reflection': reflection,
                'probe_question': pq_match.group(1).strip() if pq_match else '(unspecified)',
                'files': files}
    return {'mode': 'finalize', 'reflection': reflection or '(no reflection parsed)',
            'probe_question': '', 'files': {}}


def _build_turn_prompt(report, prior_turns, focus, context_note):
    prior_txt = "(none yet this session)" if not prior_turns else "\n\n".join(
        f"Round {t.get('round', 0)}: {t['probe_question']}\n"
        f"```json\n{json.dumps(t['result'], indent=2)}\n```"
        for t in prior_turns)
    focus_block = f"\n\nWhy this round was requested: {focus}" if focus else ''
    context_block = f"\n\nRecent run context: {context_note}" if context_note else ''
    return f"""{_schema_text()}

Deterministic EDA report (JSON):
```json
{json.dumps(report, indent=2)}
```

Every probe run so far, this project (across every round, including this one's earlier \
turns):
{prior_txt}{focus_block}{context_block}

Decide your next action now."""


def run_agentic_exploration(report, data_dir, model, max_turns, max_budget_usd,
                             max_repairs, propose_timeout, probe_timeout, out_dir,
                             focus=None, context_note=None):
    """Runs up to max_turns write-probe -> run -> see-result turns, stopping early on
    FINALIZE, budget exhaustion, or repeated failure. Appends every turn (successes and
    failures alike) to agent/runs/eda_probes/log.json under a new 'round' number, never
    overwriting prior rounds. Never raises -- worst case, zero new turns get logged and
    accumulated_findings_text() returns whatever prior rounds already found."""
    out_dir = Path(out_dir)
    probes_root = out_dir / 'eda_probes'
    probes_root.mkdir(parents=True, exist_ok=True)
    round_num = _next_round(out_dir)
    prior_turns = [t for t in _load_log(out_dir)['turns']
                   if t.get('mode') == 'probe' and 'result' in t]

    tracker = llm.BudgetTracker(max_budget_usd)
    new_turns = []
    consecutive_failures = 0
    stopped_reason = 'turn_limit'

    for i in range(max_turns):
        if tracker.exhausted():
            stopped_reason = 'budget_exhausted'
            break

        prompt = _build_turn_prompt(report, prior_turns, focus, context_note)
        attempt, tries = probe_runner.call_llm_with_retry(
            SYSTEM_PROMPT, prompt, model, max_budget_usd,
            label=f'agentic EDA turn {i} (round {round_num})', timeout=propose_timeout)
        llm_calls = [{'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')}
                     for t in tries]
        for t in tries:
            tracker.record(t.get('cost_usd', 0.0))

        if not attempt['ok']:
            new_turns.append({'round': round_num, 'turn': i, 'mode': 'failed',
                               'llm_calls': llm_calls, 'error': attempt['error']})
            stopped_reason = 'llm_call_failed'
            break

        parsed = _parse_turn(attempt['text'])
        if parsed['mode'] == 'finalize':
            new_turns.append({'round': round_num, 'turn': i, 'mode': 'finalize',
                               'reflection': parsed['reflection'], 'llm_calls': llm_calls})
            stopped_reason = 'finalize'
            break

        turn_dir = probes_root / f'round_{round_num}_turn_{i}'
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / 'probe.py').write_text(parsed['files']['probe.py'], encoding='utf-8')

        result, run_attempts = probe_runner.run_probe_and_repair(
            turn_dir, parsed['probe_question'], data_dir, model, max_budget_usd,
            propose_timeout, probe_timeout, max_repairs, llm_calls,
            system_prompt=SYSTEM_PROMPT, allowed_files=pt.PROBE_ALLOWED_FILES)
        for c in llm_calls:
            tracker.record(c.get('cost_usd', 0.0))

        if result['ok']:
            consecutive_failures = 0
            turn_record = {'round': round_num, 'turn': i, 'mode': 'probe',
                            'reflection': parsed['reflection'],
                            'probe_question': parsed['probe_question'],
                            'run_attempts': run_attempts, 'llm_calls': llm_calls,
                            'result': result['result']}
            new_turns.append(turn_record)
            prior_turns.append(turn_record)
        else:
            consecutive_failures += 1
            new_turns.append({'round': round_num, 'turn': i, 'mode': 'failed',
                               'reflection': parsed['reflection'],
                               'probe_question': parsed['probe_question'],
                               'run_attempts': run_attempts, 'llm_calls': llm_calls,
                               'error': result['error']})
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                stopped_reason = 'repeated_failure'
                break
    else:
        stopped_reason = 'turn_limit'

    log = _load_log(out_dir)
    log['turns'].extend(new_turns)
    log.setdefault('rounds', []).append({
        'round': round_num, 'focus': focus, 'stopped_reason': stopped_reason,
        'cost_usd': tracker.spent_usd, 'budget_ceiling_usd': max_budget_usd,
    })
    _log_path(out_dir).write_text(json.dumps(log, indent=2), encoding='utf-8')
    return {'round': round_num, 'stopped_reason': stopped_reason, 'cost_usd': tracker.spent_usd,
            'new_turns': new_turns}
