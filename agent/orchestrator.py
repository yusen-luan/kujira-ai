"""v2 orchestrator: a single linear iteration chain (replaces v1's feature/model
axis-tree). Each iteration looks at the whole current pipeline (data.py +
baseline.py together) and the LLM itself picks the single highest-leverage
change to make — a feature, a model/architecture change, or a training/loss
change — rather than being pre-assigned to one axis. See prompt_templates.py's
module docstring and agent_notes/orchestrator.md for the full design
discussion and why v1's axis split was replaced (it had a real bug: the model
axis never inherited the feature axis's accepted improvements, so every
model-architecture hypothesis was tested against a strictly worse feature set
than what was actually available — collapsing to one chain fixes this by
construction, since there's only ever one current-best state).

Per-iteration flow:
    1. Propose call sees the full current data.py + baseline.py, its own last
       iteration's outcome (including a parsed training-curve summary, not
       just the endpoint metrics — see parse_training_curve), EDA facts,
       literature, and accumulated probe findings. It picks exactly one of:
         (a) a hypothesis + code change (apply -> run -> accept/reject), or
         (b) a diagnostic probe question (run a small read-only computation,
             append the answer to probe_findings.md, no code change this turn).
    2. A plain consecutive-non-improvement streak drives an explicit "you're
       plateauing" instruction into the next propose prompt once it crosses
       --escalate_after — replaces v1's separate diagnosis-classification
       call with a cheaper in-prompt nudge.
    3. The official convergence rule (epsilon=--converge_eps, N=--converge_n
       consecutive nodes with no improvement) stops the run early, same as v1.
       A single epsilon (--converge_eps) drives both accept/reject and the
       plateau streak, so the two can't disagree: a node is accepted iff its
       gain over the current best exceeds converge_eps, and that's exactly
       the condition that resets the streak from step 2 to 0 -- any accepted
       node resets it, any rejected node increments it.

Usage:
    python orchestrator.py                       # reproduce baseline, then 4 LLM iterations
    python orchestrator.py --iterations 0         # just reproduce baseline (sanity check)
    python orchestrator.py --iterations 10 --max_repairs 2

State on disk:
    agent/runs/best/          current accepted data.py + baseline.py (single chain)
    agent/runs/node_N/        one node's working copy (kept after the run for inspection)
    agent/runs/probe_findings.md  accumulated diagnostic-probe results (deliverable-relevant)
    logs/node_N.json          full record of that node (deliverable #3)
"""
import argparse
import json
import queue
import re
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



def ensure_eda(args):
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
    if rag.CONTEXT_PATH.exists() and not args.regen_eda:
        display.phase(f'Literature: reusing existing retrieval in {rag.RUNS_DIR}')
        return
    display.phase('Literature: retrieving relevant corpus notes (one-time, local BM25)...')
    t0 = time.time()
    rag.run()
    display.phase(f'literature retrieval done in {time.time() - t0:.0f}s')


def ensure_best_dir():
    bd = RUNS_DIR / 'best'
    bd.mkdir(parents=True, exist_ok=True)
    for fname in pt.ALLOWED_FILES:
        dst = bd / fname
        if not dst.exists():
            shutil.copy2(WORKSPACE / fname, dst)
    return bd


def reset_best_dir():
    """--no-resume: discard any previously-accepted code changes and start this run's
    chain from workspace/'s original data.py + baseline.py, same as a first-ever run."""
    bd = RUNS_DIR / 'best'
    bd.mkdir(parents=True, exist_ok=True)
    for fname in pt.ALLOWED_FILES:
        shutil.copy2(WORKSPACE / fname, bd / fname)
    return bd


def node0_label(best_dir):
    """node 0 always just re-runs whatever's in best_dir -- with --resume (the default),
    that's the prior session's already-accepted state if one exists, not necessarily
    workspace/'s pristine files. Calling that 'baseline' is misleading (it isn't the
    organizer's reference once anything's been accepted), so compare contents and pick
    the right label rather than hardcoding 'baseline'."""
    is_pristine = all((best_dir / f).read_text(encoding='utf-8') == (WORKSPACE / f).read_text(encoding='utf-8')
                       for f in pt.ALLOWED_FILES)
    return 'baseline' if is_pristine else 'last best'


def clear_prior_logs():
    """--no-resume: remove prior logs/node_N.json (N>=1) so this run's log directory
    only ever reflects this run -- otherwise a short fresh run would leave a longer
    prior run's higher-numbered node logs behind, stale and orphaned."""
    if not LOGS_DIR.exists():
        return
    for path in LOGS_DIR.glob('node_*.json'):
        if int(path.stem.split('_', 1)[1]) >= 1:
            path.unlink()


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
    ~1s) even if the child produces no output at all. `-u` on the child's own
    invocation (see callers) keeps Python's stdout line-buffered so lines actually
    arrive as they're printed, not in bursts.
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
            break
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


_EPOCH_LINE_RE = re.compile(
    r"epoch\s*(\d+)\s*\|\s*loss\s*([\d.]+)\s*\|\s*valid GAUC\s*([\d.]+)\s*nDCG@5\s*([\d.]+)\s*primary\s*([\d.]+)")
_EARLY_STOP_RE = re.compile(r"early stop at epoch\s*(\d+)")


def parse_training_curve(output_text):
    """Extracts a compact training-curve summary from a candidate's captured stdout
    (the per-epoch lines the CONTRACT asks every candidate to print under
    verbose=True). Returns None if no epoch lines were found (e.g. run_pop/
    run_random, or a candidate that doesn't print per-epoch progress). This is what
    lets the *next* iteration's REFLECTION actually diagnose overfitting/underfitting
    instead of only seeing the endpoint metrics — see prompt_templates._format_curve."""
    epochs = []
    for m in _EPOCH_LINE_RE.finditer(output_text):
        epochs.append({'epoch': int(m.group(1)), 'loss': float(m.group(2)), 'primary': float(m.group(5))})
    if not epochs:
        return None
    best = max(epochs, key=lambda e: e['primary'])
    last = epochs[-1]
    early_stopped = bool(_EARLY_STOP_RE.search(output_text))
    return {
        'n_epochs_logged': len(epochs),
        'first_epoch_primary': epochs[0]['primary'],
        'best_epoch': best['epoch'], 'best_epoch_primary': best['primary'],
        'last_epoch': last['epoch'], 'last_epoch_primary': last['primary'],
        'early_stopped': early_stopped,
        'degraded_after_best': last['epoch'] > best['epoch'] and last['primary'] < best['primary'] - 1e-4,
        'still_improving_at_cutoff': (not early_stopped) and last['epoch'] == best['epoch'],
    }


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
    return {'ok': True, 'metrics': metrics, 'training_curve': parse_training_curve(output)}


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


def load_prior_history():
    """Replays every node_*.json already on disk from a previous run into the same
    entry shape run_iteration() returns, so a fresh run's very first propose prompt
    already has the full "History of past iterations" and last-iteration reflection
    -- otherwise each run starts amnesiac and can re-propose something already tried
    and rejected. Returns (history_entries_oldest_first, highest_node_id_seen), the
    latter used so this run's node ids continue on rather than overwriting the prior
    run's logs/node_N.json files."""
    entries = []
    max_node = 0
    if not LOGS_DIR.exists():
        return entries, max_node
    paths = sorted(LOGS_DIR.glob('node_*.json'),
                    key=lambda p: int(p.stem.split('_', 1)[1]))
    for path in paths:
        node_num = int(path.stem.split('_', 1)[1])
        max_node = max(max_node, node_num)
        if node_num == 0:
            continue  # baseline reproduction, not an iteration -- re-run fresh each time
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
        status = record.get('status')
        entry = {'iter': record.get('iter', node_num), 'status': status}
        if status in ('accepted', 'rejected'):
            entry.update(hypothesis=record.get('hypothesis'), reflection=record.get('reflection'),
                         expected_effect=record.get('expected_effect'), metrics=record.get('metrics'),
                         prev_metrics=record.get('prev_metrics'), training_curve=record.get('training_curve'),
                         primary=record.get('new_primary'), prev_best=record.get('prev_best_primary'),
                         sweep=record.get('sweep'))
        elif status == 'answered':
            entry.update(question=record.get('question'), reflection=record.get('reflection'))
        elif status == 'failed':
            entry.update(hypothesis=record.get('hypothesis'), question=record.get('question'),
                         error_summary=record.get('error_summary'))
        else:
            continue
        entries.append(entry)
    return entries, max_node


def run_and_repair(node_id, iter_dir, hypothesis, args, llm_calls, hparams_override=None):
    """Runs an already-populated candidate dir; on failure, feeds the traceback
    back to the LLM and retries up to --max_repairs times. Returns (result, run_attempts).
    hparams_override merges over the base {k, lr, epochs} -- used by run_sweep below to
    try the first of several hyperparameter values through the normal repair path (a code
    bug would affect every value identically, so it's only worth debugging once)."""
    metrics_path = iter_dir / 'metrics.json'
    hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs, **(hparams_override or {})}
    timeout = args.run_timeout

    run_attempts = []
    display.run_start(f'candidate (node {node_id})', timeout)
    t0 = time.time()
    result = run_candidate(iter_dir, args.data_dir, metrics_path, hparams, args.seed, timeout,
                            verbose=not args.quiet)
    display.run_end(t0, result['ok'], timed_out=result.get('timed_out', False))
    run_attempts.append({'attempt': 0, 'ok': result['ok'], 'error': result.get('error')})

    repairs_left = args.max_repairs
    while not result['ok'] and repairs_left > 0:
        repairs_left -= 1
        repair_prompt = pt.build_repair_prompt(read_code(iter_dir), hypothesis, result['error'])
        attempt, tries = call_llm_with_retry(pt.SYSTEM_PROMPT, repair_prompt, args.model, args.max_budget_usd,
                                              label=f'repairing candidate after run failure '
                                                    f'({repairs_left + 1} attempt(s) left)')
        llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
        if not attempt['ok']:
            run_attempts.append({'attempt': len(run_attempts), 'ok': False,
                                  'error': f'repair LLM call failed: {attempt["error"]}'})
            break
        repaired_files = pt.parse_response(attempt['text'])['files']
        apply_files(iter_dir, repaired_files)
        display.run_start(f'candidate (node {node_id}, repaired)', timeout)
        t0 = time.time()
        result = run_candidate(iter_dir, args.data_dir, metrics_path, hparams, args.seed, timeout,
                                verbose=not args.quiet)
        display.run_end(t0, result['ok'], timed_out=result.get('timed_out', False))
        run_attempts.append({'attempt': len(run_attempts), 'ok': result['ok'], 'error': result.get('error')})
    return result, run_attempts


def run_sweep(node_id, iter_dir, hypothesis, args, llm_calls, sweep_param, sweep_values):
    """One node, several hparams values of the same code -- avoids spending a full extra
    LLM round-trip (and a full extra logged/attributed node) per value the way separate
    hypothesis turns would. The first value gets the normal run_and_repair treatment (so a
    real code bug still gets fixed); later values that fail (e.g. a larger width OOMs or
    times out) are just recorded as failed and skipped -- a bad hyperparameter value isn't
    a code bug worth spending repair budget on. Returns (best_result_or_None, best_value,
    trials, run_attempts), where best_result is a run_candidate-shaped 'ok' dict (or the
    first trial's failing result if every value failed) and trials is a list of
    {'value', 'ok', 'primary', 'error'} for every value tried, in order."""
    first_result, run_attempts = run_and_repair(node_id, iter_dir, hypothesis, args, llm_calls,
                                                  hparams_override={sweep_param: sweep_values[0]})
    trials = [{'value': sweep_values[0], 'ok': first_result['ok'],
               'primary': first_result['metrics']['valid']['primary'] if first_result['ok'] else None,
               'error': None if first_result['ok'] else first_result.get('error')}]
    best_result = first_result if first_result['ok'] else None
    best_value = sweep_values[0]

    base_hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs}
    metrics_path = iter_dir / 'metrics.json'
    for value in sweep_values[1:]:
        display.run_start(f'candidate (node {node_id}, {sweep_param}={value})', args.run_timeout)
        t0 = time.time()
        r = run_candidate(iter_dir, args.data_dir, metrics_path, {**base_hparams, sweep_param: value},
                           args.seed, args.run_timeout, verbose=not args.quiet)
        display.run_end(t0, r['ok'], timed_out=r.get('timed_out', False))
        trials.append({'value': value, 'ok': r['ok'],
                        'primary': r['metrics']['valid']['primary'] if r['ok'] else None,
                        'error': None if r['ok'] else r.get('error')})
        if r['ok'] and (best_result is None or r['metrics']['valid']['primary'] > best_result['metrics']['valid']['primary']):
            best_result, best_value = r, value

    return best_result if best_result is not None else first_result, best_value, trials, run_attempts


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
        attempt, tries = call_llm_with_retry(pt.SYSTEM_PROMPT, repair_prompt, args.model, args.max_budget_usd,
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


def append_probe_finding(node_id, question, result):
    PROBE_FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    block = f"### Probe (node {node_id})\nQuestion: {question}\n\n```json\n{json.dumps(result, indent=2)}\n```\n\n"
    with open(PROBE_FINDINGS_PATH, 'a', encoding='utf-8') as f:
        f.write(block)


def run_iteration(node_id, best_dir, best_primary, prev_metrics, history, args, streak):
    """One node of the single iteration chain. Proposes; the LLM itself picks a
    hypothesis+code-change turn or a probe turn. Accepts a hypothesis into `best_dir`
    (in place) on improvement. Returns (status, new_best_primary, new_best_metrics,
    history_entry). status is one of accepted/rejected/answered/failed."""
    best_code = read_code(best_dir)
    propose_prompt = pt.build_propose_prompt(best_code, history, best_primary, streak,
                                              args.escalate_after, args.converge_eps)
    llm_calls = []

    attempt, tries = call_llm_with_retry(pt.SYSTEM_PROMPT, propose_prompt, args.model, args.max_budget_usd,
                                          label='deciding the next move (hypothesis or probe)')
    llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
    if not attempt['ok']:
        record = {'iter': node_id, 'status': 'failed', 'hypothesis': None,
                   'error_summary': f'LLM propose call failed: {attempt["error"]}',
                   'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return 'failed', best_primary, prev_metrics, {'iter': node_id, 'status': 'failed',
                                         'hypothesis': '(propose call failed)', 'error_summary': attempt['error']}

    parsed = pt.parse_response(attempt['text'])
    reflection = parsed['reflection']
    display.field('reflection', reflection)

    if parsed['mode'] == 'probe':
        question = parsed['probe_question']
        display.field('probe question', question)
        probe_files = {k: v for k, v in parsed['files'].items() if k in pt.PROBE_ALLOWED_FILES}
        if not probe_files:
            record = {'iter': node_id, 'status': 'failed', 'question': question,
                       'error_summary': 'no probe.py parsed from LLM response',
                       'raw_response': attempt['text'][:2000], 'llm_calls': llm_calls}
            write_log(node_id, record)
            return 'failed', best_primary, prev_metrics, {'iter': node_id, 'status': 'failed', 'question': question}

        probe_dir = RUNS_DIR / f'node_{node_id}'
        probe_dir.mkdir(parents=True, exist_ok=True)
        for fname, content in probe_files.items():
            (probe_dir / fname).write_text(content, encoding='utf-8')

        result, run_attempts = run_probe_and_repair(probe_dir, question, args, llm_calls)
        if not result['ok']:
            record = {'iter': node_id, 'status': 'failed', 'question': question,
                       'error_summary': result['error'], 'llm_calls': llm_calls, 'run_attempts': run_attempts}
            write_log(node_id, record)
            return 'failed', best_primary, prev_metrics, {'iter': node_id, 'status': 'failed', 'question': question}

        display.field('probe result', json.dumps(result['result']))
        append_probe_finding(node_id, question, result['result'])
        record = {'iter': node_id, 'status': 'answered', 'question': question, 'reflection': reflection,
                  'result': result['result'], 'llm_calls': llm_calls, 'run_attempts': run_attempts}
        write_log(node_id, record)
        return 'answered', best_primary, prev_metrics, {'iter': node_id, 'status': 'answered',
                                                          'question': question, 'reflection': reflection}

    # hypothesis mode
    hypothesis, expected_effect, files = parsed['hypothesis'], parsed['expected_effect'], parsed['files']
    sweep_param, sweep_values = parsed['sweep_param'], parsed['sweep_values']
    display.field('hypothesis', hypothesis)
    display.field('expected effect', expected_effect)
    if sweep_param:
        display.field('sweep', f'{sweep_param} over {sweep_values}')
    changed = [f for f in files if f in pt.ALLOWED_FILES]
    if not changed:
        record = {'iter': node_id, 'status': 'failed', 'hypothesis': hypothesis,
                   'reflection': reflection, 'expected_effect': expected_effect,
                   'error_summary': 'no valid file changes parsed from LLM response',
                   'raw_response': attempt['text'][:2000], 'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return 'failed', best_primary, prev_metrics, {'iter': node_id, 'status': 'failed',
                                         'hypothesis': hypothesis, 'error_summary': 'no file changes proposed'}

    iter_dir = snapshot_node_dir(node_id, best_dir)
    apply_files(iter_dir, files)
    sweep = None
    if sweep_param:
        result, best_value, trials, run_attempts = run_sweep(node_id, iter_dir, hypothesis, args, llm_calls,
                                                               sweep_param, sweep_values)
        sweep = {'param': sweep_param, 'trials': trials, 'best_value': best_value}
    else:
        result, run_attempts = run_and_repair(node_id, iter_dir, hypothesis, args, llm_calls)

    if not result['ok']:
        record = {'iter': node_id, 'status': 'failed', 'hypothesis': hypothesis,
                   'reflection': reflection, 'expected_effect': expected_effect, 'changed_files': changed,
                   'sweep': sweep, 'error_summary': result['error'], 'llm_calls': llm_calls,
                   'run_attempts': run_attempts}
        write_log(node_id, record)
        return 'failed', best_primary, prev_metrics, {'iter': node_id, 'status': 'failed',
                                         'hypothesis': hypothesis, 'error_summary': result['error']}

    new_primary = result['metrics']['valid']['primary']
    if new_primary > best_primary + args.converge_eps:
        for fname in pt.ALLOWED_FILES:
            shutil.copy2(iter_dir / fname, best_dir / fname)
        status, out_best, out_metrics = 'accepted', new_primary, result['metrics']
    else:
        status, out_best, out_metrics = 'rejected', best_primary, prev_metrics

    record = {'iter': node_id, 'status': status, 'hypothesis': hypothesis,
              'reflection': reflection, 'expected_effect': expected_effect, 'changed_files': changed,
              'sweep': sweep, 'metrics': result['metrics'], 'prev_metrics': prev_metrics,
              'training_curve': result.get('training_curve'),
              'prev_best_primary': best_primary, 'new_primary': new_primary,
              'llm_calls': llm_calls, 'run_attempts': run_attempts}
    write_log(node_id, record)
    return status, out_best, out_metrics, {'iter': node_id, 'status': status, 'hypothesis': hypothesis,
                               'reflection': reflection, 'expected_effect': expected_effect, 'sweep': sweep,
                               'metrics': result['metrics'], 'prev_metrics': prev_metrics,
                               'training_curve': result.get('training_curve'),
                               'primary': new_primary, 'prev_best': best_primary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(WORKSPACE / 'data'))
    ap.add_argument('--iterations', type=int, default=10, help='max LLM-proposed iterations')
    ap.add_argument('--max_repairs', type=int, default=2, help='error-repair attempts per node')
    ap.add_argument('--run_timeout', type=int, default=400,
                     help='seconds before any candidate run is killed (covers both numpy and torch candidates)')
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
    ap.add_argument('--escalate_after', type=int, default=2,
                     help='consecutive iterations without a >converge_eps improvement before the '
                          'prompt explicitly pushes for a structurally different idea (or a probe)')
    ap.add_argument('--probe_timeout', type=int, default=90,
                     help='seconds before a diagnostic probe run is killed')
    ap.add_argument('--converge_eps', type=float, default=0.002,
                     help='official convergence epsilon: stop early once no node improves the overall '
                          'best primary by more than this over --converge_n consecutive nodes')
    ap.add_argument('--converge_n', type=int, default=3,
                     help='official convergence N (see --converge_eps)')
    ap.add_argument('--early_stop', action=argparse.BooleanOptionalAction, default=True,
                     help='stop early once the official convergence rule triggers (default); '
                          '--no-early_stop runs the full --iterations budget regardless, still '
                          'logging plateau streaks and escalating the propose prompt as usual')
    ap.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                     help='start from the current agent/runs/best/ code and prior logs/node_*.json '
                          'history (default), continuing node numbering so nothing is overwritten; '
                          '--no-resume starts completely fresh -- resets best/ to workspace/\'s '
                          'original data.py+baseline.py, clears prior logs/node_N.json (N>=1), '
                          'and begins node numbering at 1 again')
    ap.add_argument('--quiet', action='store_true',
                     help='suppress live per-line training/probe subprocess output (everything else '
                          '-- LLM call previews, results, the run history -- still prints)')
    args = ap.parse_args()
    display.set_quiet(args.quiet)

    ensure_eda(args)
    ensure_literature(args)
    if args.resume:
        best_dir = ensure_best_dir()
    else:
        best_dir = reset_best_dir()
        clear_prior_logs()

    label = node0_label(best_dir)
    display.banner(f'node 0: reproducing {label}')
    hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs}
    metrics_path = RUNS_DIR / 'node_0_metrics.json'
    display.run_start(label, args.run_timeout)
    t0 = time.time()
    result = run_candidate(best_dir, args.data_dir, metrics_path, hparams, args.seed, args.run_timeout,
                            verbose=not args.quiet)
    display.run_end(t0, result['ok'])
    if not result['ok']:
        print(f'FATAL: {label} reproduction failed: {result["error"]}')
        write_log(0, {'iter': 0, 'status': 'failed', 'error_summary': result['error']})
        sys.exit(1)
    baseline_primary = result['metrics']['valid']['primary']
    write_log(0, {'iter': 0, 'status': label, 'metrics': result['metrics']})
    display.phase(f'{label} valid primary = {baseline_primary:.4f}')

    best_primary = baseline_primary
    best_metrics = result['metrics']
    if args.resume:
        history, max_prior_node = load_prior_history()
        node_id = max_prior_node + 1 if max_prior_node >= 1 else 1
    else:
        history, node_id = [], 1
    converged = False
    plateau_streak = 0  # always starts fresh, even when resuming -- only the propose-prompt
                         # history/reflection context carries over, not the escalation pressure
    if history:
        display.phase(f'loaded {len(history)} prior iteration(s) from {LOGS_DIR} '
                       f'(plateau streak reset to 0); this run continues at node {node_id}')

    def show_history():
        print()
        display.render_history(baseline_primary, history, label=label)

    for _ in range(args.iterations):
        if converged:
            break
        display.banner(f'node {node_id}')
        status, best_primary, best_metrics, entry = run_iteration(
            node_id, best_dir, best_primary, best_metrics, history, args, plateau_streak)
        history.append(entry)

        if status in ('accepted', 'rejected'):
            display.result_line(entry['status'], entry.get('hypothesis'), entry.get('primary'),
                                 entry.get('prev_best'), entry.get('error_summary'))
        elif status == 'answered':
            display.probe_line('answered', entry.get('question'))
        elif entry.get('question') is not None:
            display.probe_line('failed', entry.get('question'))
        else:
            display.result_line('failed', entry.get('hypothesis'), None, None, entry.get('error_summary'))
        node_id += 1

        if status in ('accepted', 'rejected'):
            gain = entry['primary'] - entry['prev_best']
            plateau_streak = 0 if gain > args.converge_eps else plateau_streak + 1
            if plateau_streak >= args.converge_n:
                if args.early_stop:
                    converged = True
                    display.converged(args.converge_eps, args.converge_n)
                else:
                    display.phase(f'plateau streak hit {plateau_streak} (would converge per '
                                   f'epsilon={args.converge_eps}, N={args.converge_n}) but '
                                   f'--no-early_stop is set -- continuing')
        # probe/failed nodes leave plateau_streak untouched -- no metric outcome to judge

        show_history()

    display.banner(f'done: best valid primary = {best_primary:.4f} (from {label} {baseline_primary:.4f})')
    print(f'plateau streak at end: {plateau_streak}')
    if converged:
        print(f'stopped early: converged per epsilon={args.converge_eps}, N={args.converge_n}')
    print(f'best code in {best_dir}, per-node logs in {LOGS_DIR}')


if __name__ == '__main__':
    main()
