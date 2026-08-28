"""v0 orchestrator: linear propose -> apply -> run -> evaluate -> accept/reject loop,
with explicit error recovery (retry-with-error-feedback, then rollback) on top.

Usage:
    python orchestrator.py                       # reproduce baseline, then 3 LLM iterations
    python orchestrator.py --iterations 0         # just reproduce baseline (sanity check)
    python orchestrator.py --iterations 5 --max_repairs 2

State on disk:
    agent/runs/best/       current accepted data.py + baseline.py (starts as workspace/'s)
    agent/runs/iter_N/     one iteration's working copy (kept after the run for inspection)
    logs/iter_N.json       full record of that iteration (deliverable #3)
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm
import prompt_templates as pt

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / 'workspace'
RUNS_DIR = Path(__file__).resolve().parent / 'runs'
BEST_DIR = RUNS_DIR / 'best'
LOGS_DIR = REPO_ROOT / 'logs'
RUN_AND_REPORT = Path(__file__).resolve().parent / 'run_and_report.py'

ACCEPT_EPS = 1e-4  # local accept/reject threshold — NOT the official convergence rule


def ensure_best_dir():
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    for fname in pt.ALLOWED_FILES:
        dst = BEST_DIR / fname
        if not dst.exists():
            shutil.copy2(WORKSPACE / fname, dst)


def read_code(dir_path):
    return {fname: (dir_path / fname).read_text(encoding='utf-8') for fname in pt.ALLOWED_FILES}


def snapshot_iter_dir(i):
    path = RUNS_DIR / f'iter_{i}'
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for fname in pt.ALLOWED_FILES:
        shutil.copy2(BEST_DIR / fname, path / fname)
    return path


def apply_files(dir_path, files):
    changed = []
    for fname, content in files.items():
        if fname not in pt.ALLOWED_FILES:
            continue  # LLM tried to touch a file it isn't allowed to — silently dropped
        (dir_path / fname).write_text(content, encoding='utf-8')
        changed.append(fname)
    return changed


def run_candidate(candidate_dir, data_dir, out_path, hparams, timeout):
    cmd = [
        sys.executable, str(RUN_AND_REPORT),
        '--candidate_dir', str(candidate_dir),
        '--pinned_dir', str(WORKSPACE),
        '--data_dir', str(data_dir),
        '--out', str(out_path),
        '--k', str(hparams['k']), '--lr', str(hparams['lr']),
        '--epochs', str(hparams['epochs']), '--seed', str(hparams['seed']),
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


def write_log(i, record):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f'iter_{i}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def run_iteration(i, best_primary, history, args):
    """Returns (status, new_best_primary, history_entry)."""
    best_code = read_code(BEST_DIR)
    propose_prompt = pt.build_propose_prompt(best_code, history, best_primary)
    llm_calls = []

    attempt, tries = call_llm_with_retry(pt.SYSTEM_PROMPT, propose_prompt, args.model, args.max_budget_usd)
    llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
    if not attempt['ok']:
        record = {'iter': i, 'status': 'failed', 'hypothesis': None,
                   'error_summary': f'LLM propose call failed: {attempt["error"]}',
                   'llm_calls': llm_calls, 'run_attempts': []}
        write_log(i, record)
        return 'failed', best_primary, {'iter': i, 'status': 'failed', 'hypothesis': '(propose call failed)',
                                         'error_summary': attempt['error']}

    hypothesis, files = pt.parse_response(attempt['text'])
    changed = [f for f in files if f in pt.ALLOWED_FILES]
    if not changed:
        record = {'iter': i, 'status': 'failed', 'hypothesis': hypothesis,
                   'error_summary': 'no valid file changes parsed from LLM response',
                   'raw_response': attempt['text'][:2000], 'llm_calls': llm_calls, 'run_attempts': []}
        write_log(i, record)
        return 'failed', best_primary, {'iter': i, 'status': 'failed', 'hypothesis': hypothesis,
                                         'error_summary': 'no file changes proposed'}

    iter_dir = snapshot_iter_dir(i)
    apply_files(iter_dir, files)
    metrics_path = iter_dir / 'metrics.json'
    hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs, 'seed': args.seed}

    run_attempts = []
    result = run_candidate(iter_dir, args.data_dir, metrics_path, hparams, args.run_timeout)
    run_attempts.append({'attempt': 0, 'ok': result['ok'], 'error': result.get('error')})

    repairs_left = args.max_repairs
    while not result['ok'] and repairs_left > 0:
        repairs_left -= 1
        repair_prompt = pt.build_repair_prompt(read_code(iter_dir), hypothesis, result['error'])
        attempt, tries = call_llm_with_retry(pt.SYSTEM_PROMPT, repair_prompt, args.model, args.max_budget_usd)
        llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
        if not attempt['ok']:
            run_attempts.append({'attempt': len(run_attempts), 'ok': False,
                                  'error': f'repair LLM call failed: {attempt["error"]}'})
            break
        _, repaired_files = pt.parse_response(attempt['text'])
        apply_files(iter_dir, repaired_files)
        result = run_candidate(iter_dir, args.data_dir, metrics_path, hparams, args.run_timeout)
        run_attempts.append({'attempt': len(run_attempts), 'ok': result['ok'], 'error': result.get('error')})

    if not result['ok']:
        record = {'iter': i, 'status': 'failed', 'hypothesis': hypothesis, 'changed_files': changed,
                   'error_summary': result['error'], 'llm_calls': llm_calls, 'run_attempts': run_attempts}
        write_log(i, record)
        return 'failed', best_primary, {'iter': i, 'status': 'failed', 'hypothesis': hypothesis,
                                         'error_summary': result['error']}

    new_primary = result['metrics']['valid']['primary']
    if new_primary > best_primary + ACCEPT_EPS:
        for fname in pt.ALLOWED_FILES:
            shutil.copy2(iter_dir / fname, BEST_DIR / fname)
        status = 'accepted'
        out_best = new_primary
    else:
        status = 'rejected'
        out_best = best_primary

    record = {'iter': i, 'status': status, 'hypothesis': hypothesis, 'changed_files': changed,
              'metrics': result['metrics'], 'prev_best_primary': best_primary, 'new_primary': new_primary,
              'llm_calls': llm_calls, 'run_attempts': run_attempts}
    write_log(i, record)
    return status, out_best, {'iter': i, 'status': status, 'hypothesis': hypothesis,
                               'primary': new_primary, 'prev_best': best_primary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(WORKSPACE / 'data'))
    ap.add_argument('--iterations', type=int, default=3, help='number of LLM-proposed iterations')
    ap.add_argument('--max_repairs', type=int, default=2, help='error-repair attempts per iteration')
    ap.add_argument('--run_timeout', type=int, default=180, help='seconds before a candidate run is killed')
    ap.add_argument('--model', default='sonnet')
    ap.add_argument('--max_budget_usd', type=float, default=0.50)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ensure_best_dir()

    print('=== iter 0: reproducing baseline ===')
    hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs, 'seed': args.seed}
    metrics_path = RUNS_DIR / 'iter_0_metrics.json'
    t0 = time.time()
    result = run_candidate(BEST_DIR, args.data_dir, metrics_path, hparams, args.run_timeout)
    if not result['ok']:
        print(f'FATAL: baseline reproduction failed: {result["error"]}')
        write_log(0, {'iter': 0, 'status': 'failed', 'error_summary': result['error']})
        sys.exit(1)
    best_primary = result['metrics']['valid']['primary']
    write_log(0, {'iter': 0, 'status': 'baseline', 'metrics': result['metrics']})
    print(f'  baseline valid primary = {best_primary:.4f}  ({time.time()-t0:.0f}s)')

    history = []
    total_cost = 0.0
    for i in range(1, args.iterations + 1):
        print(f'=== iter {i} ===')
        t0 = time.time()
        status, best_primary, entry = run_iteration(i, best_primary, history, args)
        history.append(entry)
        elapsed = time.time() - t0
        if status == 'accepted':
            print(f'  ACCEPTED  "{entry["hypothesis"]}"  primary {entry["primary"]:.4f} '
                  f'(was {entry["prev_best"]:.4f})  [{elapsed:.0f}s]')
        elif status == 'rejected':
            print(f'  rejected  "{entry["hypothesis"]}"  primary {entry["primary"]:.4f} '
                  f'(best stays {best_primary:.4f})  [{elapsed:.0f}s]')
        else:
            print(f'  FAILED    "{entry.get("hypothesis")}"  {entry.get("error_summary")}  [{elapsed:.0f}s]')

    print(f'\n=== done: best valid primary = {best_primary:.4f} ===')
    print(f'best code in {BEST_DIR}, per-iteration logs in {LOGS_DIR}')


if __name__ == '__main__':
    main()
