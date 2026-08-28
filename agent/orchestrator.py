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

Usage:
    python orchestrator.py                       # reproduce baseline, then 4 LLM iterations
    python orchestrator.py --iterations 0         # just reproduce baseline (sanity check)
    python orchestrator.py --iterations 6 --max_repairs 2

State on disk:
    agent/runs/best_<axis>/   current accepted data.py + baseline.py for that axis's branch
    agent/runs/best/          overall best (whichever axis, or synthesis, wins) - written at the end
    agent/runs/node_N/        one node's working copy (kept after the run for inspection)
    logs/node_N.json          full record of that node (deliverable #3)
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda
import llm
import prompt_templates as pt

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / 'workspace'
RUNS_DIR = Path(__file__).resolve().parent / 'runs'
LOGS_DIR = REPO_ROOT / 'logs'
RUN_AND_REPORT = Path(__file__).resolve().parent / 'run_and_report.py'

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
        print(f'=== EDA: reusing existing report/summary in {eda.RUNS_DIR} ===')
        return
    print('=== EDA: computing data report (one-time) ===')
    t0 = time.time()
    eda.run(args.data_dir, model=args.model, max_budget_usd=args.max_budget_usd,
            skip_llm=args.skip_eda_llm)
    print(f'  EDA done in {time.time() - t0:.0f}s')


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


def run_candidate(candidate_dir, data_dir, out_path, hparams, seed, timeout):
    cmd = [
        sys.executable, str(RUN_AND_REPORT),
        '--candidate_dir', str(candidate_dir),
        '--pinned_dir', str(WORKSPACE),
        '--data_dir', str(data_dir),
        '--out', str(out_path),
        '--hparams', json.dumps(hparams),
        '--seed', str(seed),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               encoding='utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        return {'ok': False, 'timed_out': True, 'error': f'candidate run exceeded {timeout}s (likely a hang)'}
    if proc.returncode != 0:
        return {'ok': False, 'timed_out': False, 'error': proc.stderr[-4000:] or '(no stderr captured)'}
    if not out_path.exists():
        return {'ok': False, 'timed_out': False, 'error': 'run exited 0 but wrote no metrics.json'}
    try:
        metrics = json.loads(out_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {'ok': False, 'timed_out': False, 'error': 'metrics.json was not valid JSON'}
    return {'ok': True, 'metrics': metrics}


def call_llm_with_retry(system_prompt, user_prompt, model, max_budget_usd, retries=1):
    attempt = llm.call_claude(system_prompt, user_prompt, model=model, max_budget_usd=max_budget_usd)
    tries = [attempt]
    while not attempt['ok'] and retries > 0:
        retries -= 1
        attempt = llm.call_claude(system_prompt, user_prompt, model=model, max_budget_usd=max_budget_usd)
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
    result = run_candidate(iter_dir, args.data_dir, metrics_path, hparams, args.seed, timeout)
    run_attempts.append({'attempt': 0, 'ok': result['ok'], 'error': result.get('error')})

    repairs_left = args.max_repairs
    while not result['ok'] and repairs_left > 0:
        repairs_left -= 1
        repair_prompt = pt.build_repair_prompt(read_code(iter_dir), hypothesis, result['error'])
        attempt, tries = call_llm_with_retry(system_prompt, repair_prompt, args.model, args.max_budget_usd)
        llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
        if not attempt['ok']:
            run_attempts.append({'attempt': len(run_attempts), 'ok': False,
                                  'error': f'repair LLM call failed: {attempt["error"]}'})
            break
        _, repaired_files = pt.parse_response(attempt['text'])
        apply_files(iter_dir, repaired_files)
        result = run_candidate(iter_dir, args.data_dir, metrics_path, hparams, args.seed, timeout)
        run_attempts.append({'attempt': len(run_attempts), 'ok': result['ok'], 'error': result.get('error')})
    return result, run_attempts


def run_axis_iteration(node_id, axis, best_dir, best_primary, history, args):
    """Proposes + evaluates one hypothesis on `axis`'s own branch. Accepts into
    `best_dir` (in place) on improvement. Returns (status, new_best_primary, history_entry)."""
    best_code = read_code(best_dir)
    system_prompt = pt.AXES[axis]['system_prompt']
    propose_prompt = pt.build_propose_prompt(axis, best_code, history, best_primary)
    llm_calls = []

    attempt, tries = call_llm_with_retry(system_prompt, propose_prompt, args.model, args.max_budget_usd)
    llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
    if not attempt['ok']:
        record = {'iter': node_id, 'axis': axis, 'status': 'failed', 'hypothesis': None,
                   'error_summary': f'LLM propose call failed: {attempt["error"]}',
                   'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return 'failed', best_primary, {'iter': node_id, 'axis': axis, 'status': 'failed',
                                         'hypothesis': '(propose call failed)', 'error_summary': attempt['error']}

    hypothesis, files = pt.parse_response(attempt['text'])
    changed = [f for f in files if f in pt.ALLOWED_FILES]
    if not changed:
        record = {'iter': node_id, 'axis': axis, 'status': 'failed', 'hypothesis': hypothesis,
                   'error_summary': 'no valid file changes parsed from LLM response',
                   'raw_response': attempt['text'][:2000], 'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return 'failed', best_primary, {'iter': node_id, 'axis': axis, 'status': 'failed',
                                         'hypothesis': hypothesis, 'error_summary': 'no file changes proposed'}

    iter_dir = snapshot_node_dir(node_id, best_dir)
    apply_files(iter_dir, files)
    result, run_attempts = run_and_repair(node_id, iter_dir, axis, system_prompt, hypothesis, args, llm_calls)

    if not result['ok']:
        record = {'iter': node_id, 'axis': axis, 'status': 'failed', 'hypothesis': hypothesis, 'changed_files': changed,
                   'error_summary': result['error'], 'llm_calls': llm_calls, 'run_attempts': run_attempts}
        write_log(node_id, record)
        return 'failed', best_primary, {'iter': node_id, 'axis': axis, 'status': 'failed',
                                         'hypothesis': hypothesis, 'error_summary': result['error']}

    new_primary = result['metrics']['valid']['primary']
    if new_primary > best_primary + ACCEPT_EPS:
        for fname in pt.ALLOWED_FILES:
            shutil.copy2(iter_dir / fname, best_dir / fname)
        status, out_best = 'accepted', new_primary
    else:
        status, out_best = 'rejected', best_primary

    record = {'iter': node_id, 'axis': axis, 'status': status, 'hypothesis': hypothesis, 'changed_files': changed,
              'metrics': result['metrics'], 'prev_best_primary': best_primary, 'new_primary': new_primary,
              'llm_calls': llm_calls, 'run_attempts': run_attempts}
    write_log(node_id, record)
    return status, out_best, {'iter': node_id, 'axis': axis, 'status': status, 'hypothesis': hypothesis,
                               'primary': new_primary, 'prev_best': best_primary}


def last_accepted_hypothesis(history, axis):
    for h in reversed(history):
        if h.get('axis') == axis and h.get('status') == 'accepted':
            return h['hypothesis']
    return '(baseline, no accepted change on this axis)'


def run_synthesis(node_id, feature_dir, model_dir, best_primary, history, args):
    """Combines the feature-axis and model-axis branch tips into one candidate.
    Returns (status, new_best_primary, history_entry, node_dir)."""
    feature_code, model_code = read_code(feature_dir), read_code(model_dir)
    feature_hyp = last_accepted_hypothesis(history, 'feature')
    model_hyp = last_accepted_hypothesis(history, 'model')
    system_prompt = pt.SYNTHESIS_SYSTEM_PROMPT
    prompt = pt.build_synthesis_prompt(feature_code, feature_hyp, model_code, model_hyp, history)
    llm_calls = []

    attempt, tries = call_llm_with_retry(system_prompt, prompt, args.model, args.max_budget_usd)
    llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
    if not attempt['ok']:
        record = {'iter': node_id, 'axis': 'synthesis', 'status': 'failed', 'hypothesis': None,
                   'error_summary': f'LLM synthesis call failed: {attempt["error"]}',
                   'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return ('failed', best_primary, {'iter': node_id, 'axis': 'synthesis', 'status': 'failed',
                 'hypothesis': '(synthesis call failed)', 'error_summary': attempt['error']}, None)

    hypothesis, files = pt.parse_response(attempt['text'])
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


def print_result(entry, elapsed):
    axis_tag = f'[{entry["axis"]}] ' if entry.get('axis') else ''
    status = entry['status']
    if status == 'accepted':
        print(f'  {axis_tag}ACCEPTED  "{entry["hypothesis"]}"  primary {entry["primary"]:.4f} '
              f'(was {entry["prev_best"]:.4f})  [{elapsed:.0f}s]')
    elif status == 'rejected':
        print(f'  {axis_tag}rejected  "{entry["hypothesis"]}"  primary {entry["primary"]:.4f} '
              f'(prev best {entry["prev_best"]:.4f})  [{elapsed:.0f}s]')
    else:
        print(f'  {axis_tag}FAILED    "{entry.get("hypothesis")}"  {entry.get("error_summary")}  [{elapsed:.0f}s]')


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
    args = ap.parse_args()

    ensure_eda(args)
    best_dirs = ensure_axis_dirs()

    print('=== node 0: reproducing baseline ===')
    hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs}
    metrics_path = RUNS_DIR / 'node_0_metrics.json'
    t0 = time.time()
    result = run_candidate(best_dirs['feature'], args.data_dir, metrics_path, hparams, args.seed, args.run_timeout)
    if not result['ok']:
        print(f'FATAL: baseline reproduction failed: {result["error"]}')
        write_log(0, {'iter': 0, 'axis': None, 'status': 'failed', 'error_summary': result['error']})
        sys.exit(1)
    baseline_primary = result['metrics']['valid']['primary']
    write_log(0, {'iter': 0, 'axis': None, 'status': 'baseline', 'metrics': result['metrics']})
    print(f'  baseline valid primary = {baseline_primary:.4f}  ({time.time()-t0:.0f}s)')

    best_primary = {axis: baseline_primary for axis in AXES_ORDER}
    history = []
    node_id = 1
    total = args.iterations

    # --- Phase 1: explore, one iteration per axis ---
    explore_axes = AXES_ORDER[:min(total, len(AXES_ORDER))]
    for axis in explore_axes:
        print(f'=== node {node_id} [{axis}] (explore) ===')
        t0 = time.time()
        status, best_primary[axis], entry = run_axis_iteration(node_id, axis, best_dirs[axis], best_primary[axis], history, args)
        history.append(entry)
        print_result(entry, time.time() - t0)
        node_id += 1

    both_explored = len(explore_axes) == len(AXES_ORDER)
    remaining = total - len(explore_axes)
    reserve_synthesis_slot = both_explored and remaining >= 1
    exploit_budget = remaining - 1 if reserve_synthesis_slot else remaining

    # --- Phase 2: exploit, remaining budget goes to the current leader axis ---
    for _ in range(max(exploit_budget, 0)):
        leader = max(best_primary, key=best_primary.get)
        print(f'=== node {node_id} [{leader}] (exploit) ===')
        t0 = time.time()
        status, best_primary[leader], entry = run_axis_iteration(node_id, leader, best_dirs[leader], best_primary[leader], history, args)
        history.append(entry)
        print_result(entry, time.time() - t0)
        node_id += 1

    overall_axis = max(best_primary, key=best_primary.get)
    overall_primary = best_primary[overall_axis]
    overall_dir = best_dirs[overall_axis]

    # --- Phase 3: the reserved slot — synthesize if both axes improved, else one more exploit ---
    if reserve_synthesis_slot:
        both_improved = all(best_primary[axis] > baseline_primary + ACCEPT_EPS for axis in AXES_ORDER)
        if both_improved:
            print(f'=== node {node_id} [synthesis] ===')
            t0 = time.time()
            status, new_primary, entry, synth_dir = run_synthesis(
                node_id, best_dirs['feature'], best_dirs['model'], overall_primary, history, args)
            history.append(entry)
            print_result(entry, time.time() - t0)
            node_id += 1
            if status == 'accepted':
                overall_axis, overall_primary, overall_dir = 'synthesis', new_primary, synth_dir
        else:
            leader = max(best_primary, key=best_primary.get)
            print(f'=== node {node_id} [{leader}] (exploit, no synthesis: only one axis improved) ===')
            t0 = time.time()
            status, best_primary[leader], entry = run_axis_iteration(node_id, leader, best_dirs[leader], best_primary[leader], history, args)
            history.append(entry)
            print_result(entry, time.time() - t0)
            node_id += 1
            overall_axis = max(best_primary, key=best_primary.get)
            overall_primary = best_primary[overall_axis]
            overall_dir = best_dirs[overall_axis]

    overall_best = RUNS_DIR / 'best'
    overall_best.mkdir(parents=True, exist_ok=True)
    for fname in pt.ALLOWED_FILES:
        shutil.copy2(overall_dir / fname, overall_best / fname)

    print(f'\n=== done: best valid primary = {overall_primary:.4f} (from [{overall_axis}]) ===')
    print(f'per-axis best: ' + ', '.join(f'{a}={best_primary[a]:.4f}' for a in AXES_ORDER))
    print(f'overall best code in {overall_best}, per-node logs in {LOGS_DIR}')


if __name__ == '__main__':
    main()
