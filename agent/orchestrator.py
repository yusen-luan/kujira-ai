"""v1 orchestrator: a shallow hypothesis tree branching along fixed axes
(feature engineering, model/architecture), each explored linearly on its own
branch, with a final synthesis step to combine both if they both improve on
the baseline. Error recovery (retry-with-error-feedback, then rollback) is
unchanged from v0.

Schedule (given --iterations total LLM-proposed iterations):
    1. Explore: one iteration per axis (feature, then model), each starting
       from the shared baseline.
    2. Exploit: remaining budget (minus one reserved slot, if both axes were
       explored) goes to whichever axis has the better validation primary so
       far, continuing from that axis's own branch tip.
    3. Synthesize: the reserved slot. If both axes improved over the
       baseline, one LLM call combines both branch tips into a single
       candidate. Otherwise the slot is spent on more exploitation instead.

Reflection + stuck-detection + diagnostic-probe loop (added 2026-08-29):
    - Every axis propose call now sees its own last iteration's hypothesis,
      stated expected effect, and actual result, and must write a REFLECTION
      before proposing the next hypothesis (see prompt_templates.py).
    - `axis_reject_streak` tracks consecutive rejections/failures per axis
      during the exploit phase. Once an axis crosses `--stuck_after`, the
      next node on that axis is spent on `run_diagnosis` instead of a normal
      propose: a dedicated LLM call classifies the root cause
      (data_gap / modeling_ceiling / low_diversity / engineering). Only
      data_gap triggers `run_probe`, which has the LLM write a small
      read-only numpy computation (agent/eda_probe.py's contract) to answer
      the specific question the diagnosis raised, executed against the same
      pre-loaded arrays eda.py's own report uses. Findings accumulate in
      agent/runs/probe_findings.md and feed every future propose prompt.
    - Diagnosis/probe nodes are spent from a separate `--diagnosis_budget`
      reserve, not from `--iterations` — getting stuck and investigating
      doesn't eat into the normal propose budget.
    - The official convergence rule (epsilon=`--converge_eps`, N=`--converge_n`
      consecutive nodes with no improvement) is now tracked globally
      (`overall_primary_trace`) and stops the run early, independent of
      `--iterations`.

Live terminal output (added 2026-08-29): see display.py. Every LLM call prints
before/after (what it's being asked to do, cost, duration, and a preview of
the parsed response fields); every training/probe subprocess is streamed
line-by-line live instead of captured silently (see stream_subprocess below);
a hypothesis tree is reprinted after every node so the branching structure
and each accept/reject/diagnosis is visible at a glance. --quiet suppresses
just the per-line subprocess streaming (everything else still prints).

Usage:
    python orchestrator.py                       # reproduce baseline, then 4 LLM iterations
    python orchestrator.py --iterations 0         # just reproduce baseline (sanity check)
    python orchestrator.py --iterations 6 --max_repairs 2

State on disk:
    agent/runs/best_<axis>/   current accepted data.py + baseline.py for that axis's branch
    agent/runs/best/          overall best (whichever axis, or synthesis, wins) - written at the end
    agent/runs/node_N/        one node's working copy (kept after the run for inspection)
    agent/runs/probe_findings.md  accumulated diagnostic-probe results (deliverable-relevant)
    logs/node_N.json          full record of that node (deliverable #3)
"""
import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import display
import eda
import llm
import prompt_templates as pt
import rag

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / 'workspace'
RUNS_DIR = Path(__file__).resolve().parent / 'runs'
LOGS_DIR = REPO_ROOT / 'logs'
RUN_AND_REPORT = Path(__file__).resolve().parent / 'run_and_report.py'
EDA_PROBE = Path(__file__).resolve().parent / 'eda_probe.py'
PROBE_FINDINGS_PATH = RUNS_DIR / 'probe_findings.md'

ACCEPT_EPS = 1e-4  # local accept/reject threshold — NOT the official convergence rule
AXES_ORDER = list(pt.AXES.keys())  # ['feature', 'model']


def ensure_eda(args):
    """Runs the one-time, pinned EDA pass (agent/eda.py) if its outputs aren't on disk
    yet, or if --regen_eda was passed. Must happen before the first propose call:
    prompt_templates.py reads eda_summary.md lazily at prompt-build time, so as long as
    this runs before run_axis_iteration, every propose/synthesis prompt in the run
    picks up real data-grounded facts instead of the "no EDA report found" fallback."""
    have_both = eda.REPORT_PATH.exists() and eda.SUMMARY_PATH.exists()
    if have_both and not args.regen_eda:
        display.phase(f'EDA: reusing existing report/summary in {eda.RUNS_DIR}')
        return
    display.phase('EDA: computing data report (one-time, deterministic)...')
    t0 = time.time()
    eda.run(args.data_dir, model=args.model, max_budget_usd=args.max_budget_usd,
            skip_llm=args.skip_eda_llm)
    display.phase(f'EDA done in {time.time() - t0:.0f}s')


def ensure_literature(args):
    """Runs the local BM25 retrieval pass (agent/rag.py) over agent/literature/ if
    its output isn't on disk yet, or --regen_eda was passed (the retrieval query is
    derived from eda_report.json, so it's naturally tied to the same regen flag --
    no separate --regen_literature). No LLM call and no network access here: pure
    local scoring over the pre-curated corpus. Must run after ensure_eda (needs
    eda_report.json) and before the first propose call."""
    if rag.CONTEXT_PATH.exists() and not args.regen_eda:
        display.phase(f'Literature: reusing existing retrieval in {rag.RUNS_DIR}')
        return
    display.phase('Literature: retrieving relevant corpus notes (one-time, local BM25)...')
    t0 = time.time()
    rag.run()
    display.phase(f'literature retrieval done in {time.time() - t0:.0f}s')


def ensure_axis_dirs():
    dirs = {}
    for axis in AXES_ORDER:
        bd = RUNS_DIR / f'best_{axis}'
        bd.mkdir(parents=True, exist_ok=True)
        for fname in pt.ALLOWED_FILES:
            dst = bd / fname
            if not dst.exists():
                shutil.copy2(WORKSPACE / fname, dst)
        dirs[axis] = bd
    return dirs


def read_code(dir_path):
    return {fname: (dir_path / fname).read_text(encoding='utf-8') for fname in pt.ALLOWED_FILES}


def snapshot_node_dir(node_id, source_dir):
    path = RUNS_DIR / f'node_{node_id}'
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for fname in pt.ALLOWED_FILES:
        shutil.copy2(source_dir / fname, path / fname)
    return path


def apply_files(dir_path, files):
    changed = []
    for fname, content in files.items():
        if fname not in pt.ALLOWED_FILES:
            continue  # LLM tried to touch a file it isn't allowed to — silently dropped
        (dir_path / fname).write_text(content, encoding='utf-8')
        changed.append(fname)
    return changed


def _pump_stream(pipe, q):
    """Background-thread target: pushes each line onto q, then a None sentinel once
    the pipe closes (process exited). Lets the main thread poll with a timeout
    instead of blocking forever on a hung, silent child process."""
    try:
        for line in iter(pipe.readline, ''):
            q.put(line)
    finally:
        q.put(None)
        pipe.close()


def stream_subprocess(cmd, timeout):
    """Runs cmd, streaming each stdout+stderr line live via display.run_line as it's
    produced, and enforcing timeout with real wall-clock granularity (checked every
    ~1s) even if the child produces no output at all (a hang, not just a slow print).
    `-u` on the child's own invocation (see callers) keeps Python's stdout
    line-buffered so lines actually arrive as they're printed, not in bursts.
    Returns (returncode_or_None, combined_output_text, timed_out: bool)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', bufsize=1)
    q = queue.Queue()
    reader = threading.Thread(target=_pump_stream, args=(proc.stdout, q), daemon=True)
    reader.start()

    lines = []
    deadline = time.time() + timeout
    timed_out = False
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            timed_out = True
            break
        try:
            line = q.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if line is None:
            break  # child closed stdout -- it has exited or is about to
        lines.append(line)
        display.run_line(line.rstrip('\n'))

    if timed_out:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return None, ''.join(lines), True

    proc.wait()
    return proc.returncode, ''.join(lines), False


def run_candidate(candidate_dir, data_dir, out_path, hparams, seed, timeout, verbose=True):
    cmd = [
        sys.executable, '-u', str(RUN_AND_REPORT),
        '--candidate_dir', str(candidate_dir),
        '--pinned_dir', str(WORKSPACE),
        '--data_dir', str(data_dir),
        '--out', str(out_path),
        '--hparams', json.dumps(hparams),
        '--seed', str(seed),
    ]
    if verbose:
        cmd.append('--verbose')
    returncode, output, timed_out = stream_subprocess(cmd, timeout)
    if timed_out:
        return {'ok': False, 'timed_out': True, 'error': f'candidate run exceeded {timeout}s (likely a hang)'}
    if returncode != 0:
        return {'ok': False, 'timed_out': False, 'error': output[-4000:] or '(no output captured)'}
    if not out_path.exists():
        return {'ok': False, 'timed_out': False, 'error': 'run exited 0 but wrote no metrics.json'}
    try:
        metrics = json.loads(out_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {'ok': False, 'timed_out': False, 'error': 'metrics.json was not valid JSON'}
    return {'ok': True, 'metrics': metrics}


def run_probe_candidate(probe_dir, data_dir, out_path, timeout):
    """Runs a probe.py through agent/eda_probe.py's fixed harness. Same success/failure
    contract as run_candidate above: exit 0 + valid JSON at --out = success."""
    cmd = [sys.executable, '-u', str(EDA_PROBE),
           '--probe_dir', str(probe_dir), '--data_dir', str(data_dir), '--out', str(out_path)]
    returncode, output, timed_out = stream_subprocess(cmd, timeout)
    if timed_out:
        return {'ok': False, 'error': f'probe run exceeded {timeout}s (likely a hang)'}
    if returncode != 0:
        return {'ok': False, 'error': output[-4000:] or '(no output captured)'}
    if not out_path.exists():
        return {'ok': False, 'error': 'probe exited 0 but wrote no result JSON'}
    try:
        result = json.loads(out_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {'ok': False, 'error': 'probe result was not valid JSON'}
    return {'ok': True, 'result': result}


def call_llm_with_retry(system_prompt, user_prompt, model, max_budget_usd, label, retries=1):
    t0 = display.llm_call_start(label)
    attempt = llm.call_claude(system_prompt, user_prompt, model=model, max_budget_usd=max_budget_usd)
    display.llm_call_end(t0, attempt)
    tries = [attempt]
    while not attempt['ok'] and retries > 0:
        retries -= 1
        display.retrying()
        t0 = display.llm_call_start(label)
        attempt = llm.call_claude(system_prompt, user_prompt, model=model, max_budget_usd=max_budget_usd)
        display.llm_call_end(t0, attempt)
        tries.append(attempt)
    return attempt, tries


def write_log(node_id, record):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f'node_{node_id}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def timeout_for(axis, args):
    return args.model_run_timeout if axis == 'model' else args.run_timeout


def run_and_repair(node_id, iter_dir, axis, system_prompt, hypothesis, args, llm_calls):
    """Runs an already-populated candidate dir; on failure, feeds the traceback
    back to the LLM and retries up to --max_repairs times. Returns (result, run_attempts)."""
    metrics_path = iter_dir / 'metrics.json'
    hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs}
    timeout = timeout_for(axis, args)

    run_attempts = []
    display.run_start(f'{axis} candidate (node {node_id})', timeout)
    t0 = time.time()
    result = run_candidate(iter_dir, args.data_dir, metrics_path, hparams, args.seed, timeout,
                            verbose=not args.quiet)
    display.run_end(t0, result['ok'], timed_out=result.get('timed_out', False))
    run_attempts.append({'attempt': 0, 'ok': result['ok'], 'error': result.get('error')})

    repairs_left = args.max_repairs
    while not result['ok'] and repairs_left > 0:
        repairs_left -= 1
        repair_prompt = pt.build_repair_prompt(read_code(iter_dir), hypothesis, result['error'])
        attempt, tries = call_llm_with_retry(system_prompt, repair_prompt, args.model, args.max_budget_usd,
                                              label=f'repairing {axis} candidate after run failure '
                                                    f'({repairs_left + 1} attempt(s) left)')
        llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
        if not attempt['ok']:
            run_attempts.append({'attempt': len(run_attempts), 'ok': False,
                                  'error': f'repair LLM call failed: {attempt["error"]}'})
            break
        repaired_files = pt.parse_response(attempt['text'])['files']
        apply_files(iter_dir, repaired_files)
        display.run_start(f'{axis} candidate (node {node_id}, repaired)', timeout)
        t0 = time.time()
        result = run_candidate(iter_dir, args.data_dir, metrics_path, hparams, args.seed, timeout,
                                verbose=not args.quiet)
        display.run_end(t0, result['ok'], timed_out=result.get('timed_out', False))
        run_attempts.append({'attempt': len(run_attempts), 'ok': result['ok'], 'error': result.get('error')})
    return result, run_attempts


def run_probe_and_repair(probe_dir, question, args, llm_calls):
    """Same retry-with-traceback-feedback pattern as run_and_repair, pointed at a
    probe.py instead of data.py/baseline.py. Returns (result, run_attempts)."""
    result_path = probe_dir / 'probe_result.json'
    run_attempts = []
    display.run_start('diagnostic probe', args.probe_timeout)
    t0 = time.time()
    result = run_probe_candidate(probe_dir, args.data_dir, result_path, args.probe_timeout)
    display.run_end(t0, result['ok'])
    run_attempts.append({'attempt': 0, 'ok': result['ok'], 'error': result.get('error')})

    repairs_left = args.max_repairs
    while not result['ok'] and repairs_left > 0:
        repairs_left -= 1
        current_probe = {'probe.py': (probe_dir / 'probe.py').read_text(encoding='utf-8')}
        repair_prompt = pt.build_repair_prompt(current_probe, question, result['error'])
        attempt, tries = call_llm_with_retry(pt.PROBE_SYSTEM_PROMPT, repair_prompt, args.model, args.max_budget_usd,
                                              label=f'repairing probe after run failure ({repairs_left + 1} left)')
        llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
        if not attempt['ok']:
            run_attempts.append({'attempt': len(run_attempts), 'ok': False,
                                  'error': f'repair LLM call failed: {attempt["error"]}'})
            break
        repaired_files = pt.parse_response(attempt['text'])['files']
        for fname, content in repaired_files.items():
            if fname in pt.PROBE_ALLOWED_FILES:
                (probe_dir / fname).write_text(content, encoding='utf-8')
        display.run_start('diagnostic probe (repaired)', args.probe_timeout)
        t0 = time.time()
        result = run_probe_candidate(probe_dir, args.data_dir, result_path, args.probe_timeout)
        display.run_end(t0, result['ok'])
        run_attempts.append({'attempt': len(run_attempts), 'ok': result['ok'], 'error': result.get('error')})
    return result, run_attempts


def run_axis_iteration(node_id, axis, best_dir, best_primary, prev_metrics, history, args):
    """Proposes + evaluates one hypothesis on `axis`'s own branch. Accepts into
    `best_dir` (in place) on improvement. `prev_metrics` is the full valid/test metrics
    dict of the state this attempt starts from (baseline, or the axis's last accepted
    state) -- carried into the history entry so the *next* iteration's reflection can
    show a real before/after per-metric delta, not just the current absolute numbers.
    Returns (status, new_best_primary, new_best_metrics, history_entry)."""
    best_code = read_code(best_dir)
    system_prompt = pt.AXES[axis]['system_prompt']
    propose_prompt = pt.build_propose_prompt(axis, best_code, history, best_primary)
    llm_calls = []

    attempt, tries = call_llm_with_retry(system_prompt, propose_prompt, args.model, args.max_budget_usd,
                                          label=f'proposing a hypothesis on the {axis} axis')
    llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
    if not attempt['ok']:
        record = {'iter': node_id, 'axis': axis, 'status': 'failed', 'hypothesis': None,
                   'error_summary': f'LLM propose call failed: {attempt["error"]}',
                   'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return 'failed', best_primary, prev_metrics, {'iter': node_id, 'axis': axis, 'status': 'failed',
                                         'hypothesis': '(propose call failed)', 'error_summary': attempt['error']}

    parsed = pt.parse_response(attempt['text'])
    reflection, hypothesis = parsed['reflection'], parsed['hypothesis']
    expected_effect, files = parsed['expected_effect'], parsed['files']
    display.field('reflection', reflection)
    display.field('hypothesis', hypothesis)
    display.field('expected effect', expected_effect)
    changed = [f for f in files if f in pt.ALLOWED_FILES]
    if not changed:
        record = {'iter': node_id, 'axis': axis, 'status': 'failed', 'hypothesis': hypothesis,
                   'reflection': reflection, 'expected_effect': expected_effect,
                   'error_summary': 'no valid file changes parsed from LLM response',
                   'raw_response': attempt['text'][:2000], 'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return 'failed', best_primary, prev_metrics, {'iter': node_id, 'axis': axis, 'status': 'failed',
                                         'hypothesis': hypothesis, 'error_summary': 'no file changes proposed'}

    iter_dir = snapshot_node_dir(node_id, best_dir)
    apply_files(iter_dir, files)
    result, run_attempts = run_and_repair(node_id, iter_dir, axis, system_prompt, hypothesis, args, llm_calls)

    if not result['ok']:
        record = {'iter': node_id, 'axis': axis, 'status': 'failed', 'hypothesis': hypothesis,
                   'reflection': reflection, 'expected_effect': expected_effect, 'changed_files': changed,
                   'error_summary': result['error'], 'llm_calls': llm_calls, 'run_attempts': run_attempts}
        write_log(node_id, record)
        return 'failed', best_primary, prev_metrics, {'iter': node_id, 'axis': axis, 'status': 'failed',
                                         'hypothesis': hypothesis, 'error_summary': result['error']}

    new_primary = result['metrics']['valid']['primary']
    if new_primary > best_primary + ACCEPT_EPS:
        for fname in pt.ALLOWED_FILES:
            shutil.copy2(iter_dir / fname, best_dir / fname)
        status, out_best, out_metrics = 'accepted', new_primary, result['metrics']
    else:
        status, out_best, out_metrics = 'rejected', best_primary, prev_metrics

    record = {'iter': node_id, 'axis': axis, 'status': status, 'hypothesis': hypothesis,
              'reflection': reflection, 'expected_effect': expected_effect, 'changed_files': changed,
              'metrics': result['metrics'], 'prev_metrics': prev_metrics,
              'prev_best_primary': best_primary, 'new_primary': new_primary,
              'llm_calls': llm_calls, 'run_attempts': run_attempts}
    write_log(node_id, record)
    return status, out_best, out_metrics, {'iter': node_id, 'axis': axis, 'status': status, 'hypothesis': hypothesis,
                               'reflection': reflection, 'expected_effect': expected_effect,
                               'metrics': result['metrics'], 'prev_metrics': prev_metrics,
                               'primary': new_primary, 'prev_best': best_primary}


def last_accepted_hypothesis(history, axis):
    for h in reversed(history):
        if h.get('axis') == axis and h.get('status') == 'accepted':
            return h['hypothesis']
    return '(baseline, no accepted change on this axis)'


def axis_reject_streak(history, axis):
    """Consecutive rejected/failed entries at the tail of `axis`'s own history (an
    'accepted' entry resets it to 0; diagnosis/probe entries are tagged
    '<axis>_diagnosis'/'<axis>_probe' so they're skipped here, not counted and not
    treated as a break)."""
    streak = 0
    for h in reversed(history):
        if h.get('axis') != axis:
            continue
        if h.get('status') == 'accepted':
            break
        if h.get('status') in ('rejected', 'failed'):
            streak += 1
        else:
            break
    return streak


def check_converged(trace, eps, n):
    """Official convergence rule: no improvement > eps over the last n nodes."""
    if len(trace) <= n:
        return False
    return (trace[-1] - trace[-1 - n]) <= eps


def append_probe_finding(node_id, axis, question, result):
    PROBE_FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    block = (f"### Probe (node {node_id}, {axis} axis)\n"
             f"Question: {question}\n\n"
             f"```json\n{json.dumps(result, indent=2)}\n```\n\n")
    with open(PROBE_FINDINGS_PATH, 'a', encoding='utf-8') as f:
        f.write(block)


def run_diagnosis(node_id, axis, history, args):
    """Classifies why `axis` has been racking up rejections. Returns
    (category, rationale, probe_question, history_entry)."""
    prompt = pt.build_diagnosis_prompt(axis, history)
    llm_calls = []
    attempt, tries = call_llm_with_retry(pt.DIAGNOSIS_SYSTEM_PROMPT, prompt, args.model, args.max_budget_usd,
                                          label=f'diagnosing why the {axis} axis is stuck')
    llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)

    diag_axis = f'{axis}_diagnosis'
    if not attempt['ok']:
        record = {'iter': node_id, 'axis': diag_axis, 'status': 'failed',
                   'error_summary': f'LLM diagnosis call failed: {attempt["error"]}', 'llm_calls': llm_calls}
        write_log(node_id, record)
        return 'engineering', f'(diagnosis call failed: {attempt["error"]})', '', \
            {'iter': node_id, 'axis': diag_axis, 'status': 'failed'}

    parsed = pt.parse_diagnosis_response(attempt['text'])
    display.field('category', parsed['category'])
    display.field('rationale', parsed['rationale'])
    display.field('probe question', parsed['probe_question'])
    record = {'iter': node_id, 'axis': diag_axis, 'status': 'diagnosed', 'category': parsed['category'],
              'rationale': parsed['rationale'], 'probe_question': parsed['probe_question'],
              'llm_calls': llm_calls}
    write_log(node_id, record)
    entry = {'iter': node_id, 'axis': diag_axis, 'status': 'diagnosed', 'category': parsed['category']}
    return parsed['category'], parsed['rationale'], parsed['probe_question'], entry


def run_probe(node_id, axis, question, args):
    """Has the LLM write + run a read-only diagnostic probe answering `question`.
    Appends the result to probe_findings.md on success. Returns a history_entry dict."""
    probe_axis = f'{axis}_probe'
    prompt = pt.build_probe_prompt(question, axis)
    llm_calls = []
    attempt, tries = call_llm_with_retry(pt.PROBE_SYSTEM_PROMPT, prompt, args.model, args.max_budget_usd,
                                          label=f'writing a diagnostic probe for the {axis} axis')
    llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
    if not attempt['ok']:
        record = {'iter': node_id, 'axis': probe_axis, 'status': 'failed', 'question': question,
                   'error_summary': f'LLM probe call failed: {attempt["error"]}', 'llm_calls': llm_calls}
        write_log(node_id, record)
        return {'iter': node_id, 'axis': probe_axis, 'status': 'failed', 'question': question}

    parsed = pt.parse_response(attempt['text'])
    files = {k: v for k, v in parsed['files'].items() if k in pt.PROBE_ALLOWED_FILES}
    if not files:
        record = {'iter': node_id, 'axis': probe_axis, 'status': 'failed', 'question': question,
                   'error_summary': 'no probe.py parsed from LLM response',
                   'raw_response': attempt['text'][:2000], 'llm_calls': llm_calls}
        write_log(node_id, record)
        return {'iter': node_id, 'axis': probe_axis, 'status': 'failed', 'question': question}

    probe_dir = RUNS_DIR / f'node_{node_id}'
    probe_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (probe_dir / fname).write_text(content, encoding='utf-8')

    result, run_attempts = run_probe_and_repair(probe_dir, question, args, llm_calls)
    if not result['ok']:
        record = {'iter': node_id, 'axis': probe_axis, 'status': 'failed', 'question': question,
                   'error_summary': result['error'], 'llm_calls': llm_calls, 'run_attempts': run_attempts}
        write_log(node_id, record)
        return {'iter': node_id, 'axis': probe_axis, 'status': 'failed', 'question': question}

    display.field('probe result', json.dumps(result['result']))
    append_probe_finding(node_id, axis, question, result['result'])
    record = {'iter': node_id, 'axis': probe_axis, 'status': 'answered', 'question': question,
              'result': result['result'], 'llm_calls': llm_calls, 'run_attempts': run_attempts}
    write_log(node_id, record)
    return {'iter': node_id, 'axis': probe_axis, 'status': 'answered', 'question': question}


def run_synthesis(node_id, feature_dir, model_dir, best_primary, history, args):
    """Combines the feature-axis and model-axis branch tips into one candidate.
    Returns (status, new_best_primary, history_entry, node_dir)."""
    feature_code, model_code = read_code(feature_dir), read_code(model_dir)
    feature_hyp = last_accepted_hypothesis(history, 'feature')
    model_hyp = last_accepted_hypothesis(history, 'model')
    system_prompt = pt.SYNTHESIS_SYSTEM_PROMPT
    prompt = pt.build_synthesis_prompt(feature_code, feature_hyp, model_code, model_hyp, history)
    llm_calls = []

    attempt, tries = call_llm_with_retry(system_prompt, prompt, args.model, args.max_budget_usd,
                                          label='synthesizing the feature-axis + model-axis candidates')
    llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
    if not attempt['ok']:
        record = {'iter': node_id, 'axis': 'synthesis', 'status': 'failed', 'hypothesis': None,
                   'error_summary': f'LLM synthesis call failed: {attempt["error"]}',
                   'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return ('failed', best_primary, {'iter': node_id, 'axis': 'synthesis', 'status': 'failed',
                 'hypothesis': '(synthesis call failed)', 'error_summary': attempt['error']}, None)

    parsed = pt.parse_response(attempt['text'])
    hypothesis, files = parsed['hypothesis'], parsed['files']
    display.field('hypothesis', hypothesis)
    changed = [f for f in files if f in pt.ALLOWED_FILES]

    node_dir = RUNS_DIR / f'node_{node_id}'
    if node_dir.exists():
        shutil.rmtree(node_dir)
    node_dir.mkdir(parents=True)
    # Seed with feature axis's data.py + model axis's baseline.py as the fallback for
    # whichever file the LLM doesn't rewrite (it's told to output full files for both
    # when merging, but this keeps a sane default if it only touches one).
    shutil.copy2(feature_dir / 'data.py', node_dir / 'data.py')
    shutil.copy2(model_dir / 'baseline.py', node_dir / 'baseline.py')

    if not changed:
        record = {'iter': node_id, 'axis': 'synthesis', 'status': 'failed', 'hypothesis': hypothesis,
                   'error_summary': 'no valid file changes parsed from LLM response',
                   'raw_response': attempt['text'][:2000], 'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return ('failed', best_primary, {'iter': node_id, 'axis': 'synthesis', 'status': 'failed',
                 'hypothesis': hypothesis, 'error_summary': 'no file changes proposed'}, None)

    apply_files(node_dir, files)
    result, run_attempts = run_and_repair(node_id, node_dir, 'synthesis', system_prompt, hypothesis, args, llm_calls)

    if not result['ok']:
        record = {'iter': node_id, 'axis': 'synthesis', 'status': 'failed', 'hypothesis': hypothesis,
                   'changed_files': changed, 'error_summary': result['error'],
                   'llm_calls': llm_calls, 'run_attempts': run_attempts}
        write_log(node_id, record)
        return ('failed', best_primary, {'iter': node_id, 'axis': 'synthesis', 'status': 'failed',
                 'hypothesis': hypothesis, 'error_summary': result['error']}, None)

    new_primary = result['metrics']['valid']['primary']
    accepted = new_primary > best_primary + ACCEPT_EPS
    status = 'accepted' if accepted else 'rejected'
    out_best = new_primary if accepted else best_primary

    record = {'iter': node_id, 'axis': 'synthesis', 'status': status, 'hypothesis': hypothesis,
              'changed_files': changed, 'metrics': result['metrics'], 'prev_best_primary': best_primary,
              'new_primary': new_primary, 'llm_calls': llm_calls, 'run_attempts': run_attempts}
    write_log(node_id, record)
    return (status, out_best, {'iter': node_id, 'axis': 'synthesis', 'status': status, 'hypothesis': hypothesis,
             'primary': new_primary, 'prev_best': best_primary}, node_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(WORKSPACE / 'data'))
    ap.add_argument('--iterations', type=int, default=4,
                     help='total LLM-proposed iterations across explore/exploit/synthesis')
    ap.add_argument('--max_repairs', type=int, default=2, help='error-repair attempts per node')
    ap.add_argument('--run_timeout', type=int, default=180, help='seconds before a feature-axis run is killed')
    ap.add_argument('--model_run_timeout', type=int, default=400,
                     help='seconds before a model-axis (torch, CPU) run is killed')
    ap.add_argument('--model', default='sonnet')
    ap.add_argument('--max_budget_usd', type=float, default=0.50)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--regen_eda', action='store_true',
                     help='recompute the EDA report/summary even if already on disk')
    ap.add_argument('--skip_eda_llm', action='store_true',
                     help='compute the EDA report but skip the LLM summarization call (no cost)')
    ap.add_argument('--stuck_after', type=int, default=2,
                     help='consecutive rejected/failed iterations on an axis (during exploit) before '
                          'spending a node on diagnosis instead of another normal propose')
    ap.add_argument('--diagnosis_budget', type=int, default=2,
                     help='extra nodes (diagnosis + probe calls) reserved outside --iterations, spent '
                          'only when an axis actually gets stuck')
    ap.add_argument('--probe_timeout', type=int, default=90,
                     help='seconds before a diagnostic probe run is killed')
    ap.add_argument('--converge_eps', type=float, default=0.002,
                     help='official convergence epsilon: stop early once no node improves the overall '
                          'best primary by more than this over --converge_n consecutive nodes')
    ap.add_argument('--converge_n', type=int, default=3,
                     help='official convergence N (see --converge_eps)')
    ap.add_argument('--quiet', action='store_true',
                     help='suppress live per-line training/probe subprocess output (everything else '
                          '-- LLM call previews, results, the hypothesis tree -- still prints)')
    args = ap.parse_args()
    display.set_quiet(args.quiet)

    ensure_eda(args)
    ensure_literature(args)
    best_dirs = ensure_axis_dirs()

    display.banner('node 0: reproducing baseline')
    hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs}
    metrics_path = RUNS_DIR / 'node_0_metrics.json'
    display.run_start('baseline (feature-axis seed)', args.run_timeout)
    t0 = time.time()
    result = run_candidate(best_dirs['feature'], args.data_dir, metrics_path, hparams, args.seed, args.run_timeout,
                            verbose=not args.quiet)
    display.run_end(t0, result['ok'])
    if not result['ok']:
        print(f'FATAL: baseline reproduction failed: {result["error"]}')
        write_log(0, {'iter': 0, 'axis': None, 'status': 'failed', 'error_summary': result['error']})
        sys.exit(1)
    baseline_primary = result['metrics']['valid']['primary']
    write_log(0, {'iter': 0, 'axis': None, 'status': 'baseline', 'metrics': result['metrics']})
    display.phase(f'baseline valid primary = {baseline_primary:.4f}')

    best_primary = {axis: baseline_primary for axis in AXES_ORDER}
    best_metrics = {axis: result['metrics'] for axis in AXES_ORDER}
    history = []
    node_id = 1
    total = args.iterations
    overall_primary_trace = [baseline_primary]
    converged = False
    diagnosis_budget_left = args.diagnosis_budget

    def note_converged():
        nonlocal converged
        overall_primary_trace.append(max(best_primary.values()))
        if not converged and check_converged(overall_primary_trace, args.converge_eps, args.converge_n):
            converged = True
            display.converged(args.converge_eps, args.converge_n)

    def show_tree():
        print()
        display.render_tree(baseline_primary, AXES_ORDER, history)

    # --- Phase 1: explore, one iteration per axis ---
    explore_axes = AXES_ORDER[:min(total, len(AXES_ORDER))]
    for axis in explore_axes:
        if converged:
            break
        display.banner(f'node {node_id} [{axis}] (explore)')
        status, best_primary[axis], best_metrics[axis], entry = run_axis_iteration(
            node_id, axis, best_dirs[axis], best_primary[axis], best_metrics[axis], history, args)
        history.append(entry)
        display.result_line(entry['status'], entry.get('hypothesis'), entry.get('primary'),
                             entry.get('prev_best'), entry.get('error_summary'))
        node_id += 1
        note_converged()
        show_tree()

    both_explored = len(explore_axes) == len(AXES_ORDER)
    remaining = total - len(explore_axes)
    reserve_synthesis_slot = both_explored and remaining >= 1
    exploit_budget = remaining - 1 if reserve_synthesis_slot else remaining

    # --- Phase 2: exploit, remaining budget goes to the current leader axis. A leader
    # axis that racks up --stuck_after consecutive rejects/fails gets a diagnosis node
    # (and, if data_gap, a probe node) instead of a normal propose — spent from the
    # separate diagnosis_budget reserve, not from exploit_budget. ---
    exploit_remaining = max(exploit_budget, 0)
    diagnosed_axes_this_streak = set()
    while exploit_remaining > 0 and not converged:
        leader = max(best_primary, key=best_primary.get)
        streak = axis_reject_streak(history, leader)
        if streak >= args.stuck_after and diagnosis_budget_left > 0 and leader not in diagnosed_axes_this_streak:
            display.banner(f'node {node_id} [{leader}] (diagnosis: {streak} consecutive non-improvements)')
            category, rationale, probe_question, entry = run_diagnosis(node_id, leader, history, args)
            history.append(entry)
            display.diagnosis_line(category, rationale)
            node_id += 1
            diagnosis_budget_left -= 1
            diagnosed_axes_this_streak.add(leader)  # don't re-diagnose until an accept resets the streak
            show_tree()
            if category == 'data_gap' and probe_question and diagnosis_budget_left > 0:
                display.banner(f'node {node_id} [{leader}] (diagnostic probe)')
                probe_entry = run_probe(node_id, leader, probe_question, args)
                history.append(probe_entry)
                display.probe_line(probe_entry['status'], probe_question)
                node_id += 1
                diagnosis_budget_left -= 1
                show_tree()
            continue  # diagnosis/probe nodes don't consume exploit_remaining

        display.banner(f'node {node_id} [{leader}] (exploit)')
        status, best_primary[leader], best_metrics[leader], entry = run_axis_iteration(
            node_id, leader, best_dirs[leader], best_primary[leader], best_metrics[leader], history, args)
        history.append(entry)
        display.result_line(entry['status'], entry.get('hypothesis'), entry.get('primary'),
                             entry.get('prev_best'), entry.get('error_summary'))
        node_id += 1
        exploit_remaining -= 1
        if status == 'accepted':
            diagnosed_axes_this_streak.discard(leader)
        note_converged()
        show_tree()

    overall_axis = max(best_primary, key=best_primary.get)
    overall_primary = best_primary[overall_axis]
    overall_dir = best_dirs[overall_axis]

    # --- Phase 3: the reserved slot — synthesize if both axes improved, else one more exploit ---
    if reserve_synthesis_slot and not converged:
        both_improved = all(best_primary[axis] > baseline_primary + ACCEPT_EPS for axis in AXES_ORDER)
        if both_improved:
            display.banner(f'node {node_id} [synthesis]')
            status, new_primary, entry, synth_dir = run_synthesis(
                node_id, best_dirs['feature'], best_dirs['model'], overall_primary, history, args)
            history.append(entry)
            display.result_line(entry['status'], entry.get('hypothesis'), entry.get('primary'),
                                 entry.get('prev_best'), entry.get('error_summary'))
            node_id += 1
            show_tree()
            if status == 'accepted':
                overall_axis, overall_primary, overall_dir = 'synthesis', new_primary, synth_dir
        else:
            leader = max(best_primary, key=best_primary.get)
            display.banner(f'node {node_id} [{leader}] (exploit, no synthesis: only one axis improved)')
            status, best_primary[leader], best_metrics[leader], entry = run_axis_iteration(
                node_id, leader, best_dirs[leader], best_primary[leader], best_metrics[leader], history, args)
            history.append(entry)
            display.result_line(entry['status'], entry.get('hypothesis'), entry.get('primary'),
                                 entry.get('prev_best'), entry.get('error_summary'))
            node_id += 1
            show_tree()
            overall_axis = max(best_primary, key=best_primary.get)
            overall_primary = best_primary[overall_axis]
            overall_dir = best_dirs[overall_axis]

    overall_best = RUNS_DIR / 'best'
    overall_best.mkdir(parents=True, exist_ok=True)
    for fname in pt.ALLOWED_FILES:
        shutil.copy2(overall_dir / fname, overall_best / fname)

    display.banner(f'done: best valid primary = {overall_primary:.4f} (from [{overall_axis}])')
    print('per-axis best: ' + ', '.join(f'{a}={best_primary[a]:.4f}' for a in AXES_ORDER))
    if converged:
        print(f'stopped early: converged per epsilon={args.converge_eps}, N={args.converge_n}')
    print(f'diagnosis/probe nodes used: {args.diagnosis_budget - diagnosis_budget_left}/{args.diagnosis_budget}')
    print(f'overall best code in {overall_best}, per-node logs in {LOGS_DIR}')


if __name__ == '__main__':
    main()
