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
    agent/runs/eda_probes/    every agentic-EDA probe ever run (bootstrap + mid-run rounds)
    logs/node_N.json          full record of that node (deliverable #3)
"""
import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import code_agent
import display
import eda
import eda_agent
import probe_runner
import prompt_templates as pt
import rag
import repo_explore
import web_research

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / 'workspace'
RUNS_DIR = Path(__file__).resolve().parent / 'runs'
LOGS_DIR = REPO_ROOT / 'logs'
RUN_AND_REPORT = Path(__file__).resolve().parent / 'run_and_report.py'
PROBE_FINDINGS_PATH = RUNS_DIR / 'probe_findings.md'

# v4 roadmap Phase 3b -- the only two packages prompt_templates.py's contract already tells the
# LLM are sanctioned/expected in baseline.py ("No other new pip dependencies" is already the
# rule), so this is exactly what's safe to auto-install unattended: nothing broader than what's
# already contractually permitted.
PIP_INSTALL_ALLOWLIST = ('torch', 'torchfm')


def preflight_check(args):
    """v4 roadmap Phase 3a, hardened: torch/torchfm are now a HARD requirement, not an optional
    fallback. Node 9 (see agent_notes/v4_roadmap.md) crashed twice on
    `ModuleNotFoundError: No module named 'torch'` -- torch was installed somewhere on the
    machine but not in the exact interpreter agent/run_and_report.py's subprocess actually uses
    (sys.executable). The original version of this function only detected that and fell back to
    numpy-only for the whole run (adjusting the propose/repair prompt accordingly) -- but that
    meant a torch-capable environment that simply never had `pip install torch torchfm` run in
    THIS interpreter silently downgraded every run to numpy-only forever, capping the
    model_architecture lever at what the LLM could hand-derive backprop for blind, in one shot,
    with zero execution feedback -- a real, discovered-in-practice ceiling on run quality, not a
    hypothetical one. Since PIP_INSTALL_ALLOWLIST above already exists specifically because
    torch/torchfm are the only two packages the propose/repair contract sanctions, self-installing
    them proactively here (before any LLM call, in the same interpreter run_and_report.py's
    subprocess will use) is strictly narrower than what's already contractually permitted --
    there's no reason to only self-heal reactively, on an in-flight crash, when we could just
    ensure the precondition up front. Now unconditional: if the install itself fails (no network,
    a broken index, disk space), preflight hard-fails the whole run rather than ever proceeding
    numpy-only -- there is no more --require_torch opt-in, because there is no more optional path
    to opt out of."""
    display.phase('Preflight: checking torch/torchfm import in the run_and_report.py subprocess interpreter...')
    proc = subprocess.run([sys.executable, '-c', 'import torch, torchfm'],
                           capture_output=True, text=True, timeout=30)
    if proc.returncode == 0:
        args.torch_available = True
        display.phase('Preflight: torch/torchfm import OK.')
        return

    diagnostic = (proc.stderr or '').strip().splitlines()[-1:] or ['(no stderr captured)']
    display.phase(f'Preflight: torch/torchfm NOT importable in {sys.executable} -- {diagnostic[0]} -- '
                  f'installing from the allowlist ({", ".join(PIP_INSTALL_ALLOWLIST)}) before proceeding, '
                  f'since torch is now a hard requirement...')
    for package in PIP_INSTALL_ALLOWLIST:
        ok, output = pip_install(package)
        display.phase(f'Preflight: pip install {package} -- {"OK" if ok else "FAILED"}')
        if not ok:
            print(f'FATAL: preflight install of {package!r} into {sys.executable} failed and torch is a '
                  f'hard requirement -- no numpy-only fallback is available anymore.\n{output[-2000:]}\n'
                  f'Fix manually: "{sys.executable}" -m pip install {" ".join(PIP_INSTALL_ALLOWLIST)}')
            sys.exit(1)

    proc = subprocess.run([sys.executable, '-c', 'import torch, torchfm'],
                           capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        diagnostic = (proc.stderr or '').strip().splitlines()[-1:] or ['(no stderr captured)']
        print(f'FATAL: torch/torchfm still not importable in {sys.executable} after installing -- '
              f'{diagnostic[0]}\n{proc.stderr[-2000:]}\n'
              f'Fix manually: "{sys.executable}" -m pip install {" ".join(PIP_INSTALL_ALLOWLIST)}')
        sys.exit(1)
    args.torch_available = True
    display.phase('Preflight: torch/torchfm installed and import OK.')


def ensure_eda(args):
    have_both = eda.REPORT_PATH.exists() and eda.SUMMARY_PATH.exists()
    if have_both and not args.regen_eda:
        display.phase(f'EDA: reusing existing report/summary in {eda.RUNS_DIR}')
        return
    display.phase('EDA: computing data report (one-time, deterministic)...')
    t0 = time.time()
    eda.run(args.data_dir, model=args.model, max_budget_usd=args.max_budget_usd,
            skip_llm=args.skip_eda_llm,
            agent_turns=(0 if args.skip_eda_agent else args.eda_agent_turns),
            agent_max_budget_usd=args.eda_agent_max_budget_usd,
            agent_max_repairs=args.max_repairs)
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


LOGS_BACKUP_DIR = LOGS_DIR / 'backup'
RUNS_BACKUP_DIR = RUNS_DIR / 'backup'


def archive_prior_run():
    """--no-resume used to just delete the previous run's state (reset_best_dir() +
    clear_prior_logs() below), which silently destroyed it -- easy to do by accident
    when reaching for --no-resume specifically to get a clean pristine-baseline test.
    Now: move everything --no-resume is about to reset into timestamped backup folders
    instead of deleting it, so a fresh test run never costs you the previous run's
    results. Both logs/ and agent/runs/ get their OWN backup/<timestamp>/ folder
    (rather than nesting one inside the other) so each stays organized under the
    directory it actually belongs to; both are already fully covered by .gitignore
    (`logs/*` / `agent/runs/`), so nothing extra needs to be added there.

    logs/backup/<timestamp>/       <- every logs/node_*.json (including node_0.json)
    agent/runs/backup/<timestamp>/ <- best/, node_*/, node_0_metrics.json,
                                       probe_findings.md, web_literature/,
                                       literature_context.md

    Deliberately does NOT touch agent/runs/{eda_report.json, eda_summary.md} -- those
    are axis-agnostic and expensive-ish to regenerate (one LLM call), not part of what
    a "fresh run" means to reset (see ensure_eda()). literature_context.md IS moved
    despite being similarly cheap-to-regenerate, specifically because a truly fresh/
    pristine run's literature grounding shouldn't silently carry over web-research
    notes found by the run being archived away -- ensure_literature() regenerates it
    from agent/literature/ alone (cheap, local BM25, no LLM cost) the moment it finds
    the cached file gone.

    Returns (logs_backup_dir, runs_backup_dir), or (None, None) if there was nothing
    worth archiving (a genuinely first-ever run)."""
    prior_node_logs = sorted(LOGS_DIR.glob('node_*.json')) if LOGS_DIR.exists() else []
    have_real_iterations = any(p.stem != 'node_0' for p in prior_node_logs)
    best_path = RUNS_DIR / 'best'
    # best/ always exists after any run, pristine or not -- checking mere existence would
    # never let this be a no-op on an already-fresh state, so check divergence instead,
    # the same way node0_label() decides whether best/ still matches workspace/.
    have_diverged_best = best_path.exists() and node0_label(best_path) != 'baseline'
    if not have_real_iterations and not have_diverged_best:
        return None, None

    stamp = time.strftime('%Y%m%d_%H%M%S')
    logs_dest = LOGS_BACKUP_DIR / stamp
    runs_dest = RUNS_BACKUP_DIR / stamp

    if prior_node_logs:
        logs_dest.mkdir(parents=True, exist_ok=True)
        for p in prior_node_logs:
            shutil.move(str(p), str(logs_dest / p.name))

    to_move = ['best', 'node_0_metrics.json', 'probe_findings.md', 'web_literature',
               'literature_context.md']
    to_move += [p.name for p in RUNS_DIR.glob('node_*') if p.is_dir()]
    to_move = [name for name in to_move if (RUNS_DIR / name).exists()]
    if to_move:
        runs_dest.mkdir(parents=True, exist_ok=True)
        for name in to_move:
            shutil.move(str(RUNS_DIR / name), str(runs_dest / name))

    return (logs_dest if prior_node_logs else None), (runs_dest if to_move else None)


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
    returncode, output, timed_out = probe_runner.stream_subprocess(cmd, timeout)
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
                         expected_effect=record.get('expected_effect'), lever_category=record.get('lever_category'),
                         metrics=record.get('metrics'),
                         prev_metrics=record.get('prev_metrics'), training_curve=record.get('training_curve'),
                         primary=record.get('new_primary'), prev_best=record.get('prev_best_primary'),
                         sweep=record.get('sweep'), variants=record.get('variants'),
                         winning_variant=record.get('winning_variant'), authored_via=record.get('authored_via'))
        elif status == 'answered':
            entry.update(question=record.get('question'), reflection=record.get('reflection'))
        elif status == 'failed':
            entry.update(hypothesis=record.get('hypothesis'), question=record.get('question'),
                         error_summary=record.get('error_summary'), lever_category=record.get('lever_category'),
                         variants=record.get('variants'), authored_via=record.get('authored_via'))
        elif status == 'researched':
            entry.update(research_question=record.get('research_question'),
                         reflection=record.get('reflection'),
                         note_title=(record.get('note') or {}).get('title'))
        elif status in ('research_failed', 'research_denied'):
            entry.update(research_question=record.get('research_question'),
                         reflection=record.get('reflection'),
                         rejected_reason=record.get('rejected_reason'))
        elif status == 'explored':
            # Was missing before this fix -- explore-turn entries silently vanished from
            # history across a --resume (the note itself still persisted via rag's
            # corpus, but "History of past iterations" and the last-turn reflection
            # check both lost the entry). Mirrors the researched/research_* handling above.
            entry.update(explore_question=record.get('explore_question'),
                         reflection=record.get('reflection'),
                         note_title=(record.get('note') or {}).get('title'))
        elif status in ('explore_failed', 'explore_denied'):
            entry.update(explore_question=record.get('explore_question'),
                         reflection=record.get('reflection'),
                         rejected_reason=record.get('rejected_reason'))
        elif status == 'eda_round':
            entry.update(eda_round_question=record.get('eda_round_question'),
                         reflection=record.get('reflection'))
        elif status in ('eda_round_failed', 'eda_round_denied'):
            entry.update(eda_round_question=record.get('eda_round_question'),
                         reflection=record.get('reflection'))
        elif status == 'code_session_denied':
            # v4 roadmap Phase 4 -- mirrors research_denied/explore_denied/eda_round_denied
            # above. A code session's ATTEMPTED outcome (accepted/rejected/failed) is already
            # covered by the generic branches above (authored_via is what marks it as a code
            # session, not the status string) -- only the denied-before-attempt case needs its
            # own branch here.
            entry.update(code_session_question=record.get('code_session_question'),
                         reflection=record.get('reflection'))
        else:
            continue
        entries.append(entry)
    return entries, max_node


_MODULE_NOT_FOUND_RE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")


def _detect_missing_package(error_text):
    """v4 roadmap Phase 3b. Scoped to exactly this one exception type/message shape (the
    confirmed node 9 signature -- see logs/node_9.json's run_attempts[0].error) rather than a
    broader ImportError catch-all: a generic ImportError (e.g. a version-mismatch `cannot import
    name X`) isn't reliably fixed by reinstalling, and pip could even make it worse by silently
    upgrading a pinned version. Returns the top-level installable package name (a submodule
    failure like `torchfm.layer` maps to 'torchfm') or None."""
    if not error_text:
        return None
    m = _MODULE_NOT_FOUND_RE.search(error_text)
    return m.group(1).split('.')[0] if m else None


_TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu'


def pip_install(package, timeout=600):
    """v4 roadmap Phase 3b -- installs into the exact same interpreter
    agent/run_and_report.py's subprocess uses (sys.executable), which is precisely what would
    have fixed node 9's actual root cause (torch present on the machine, absent from the right
    interpreter). Only ever called with a name already checked against PIP_INSTALL_ALLOWLIST by
    the caller. Reuses probe_runner.stream_subprocess for the same live-streamed-output + real
    wall-clock timeout behavior run_candidate() itself relies on. Returns (ok, output).

    Two environment-specific fixes discovered running preflight_check() for real (see its
    docstring): (1) `--break-system-packages` -- Debian/Ubuntu system Pythons (PEP 668) refuse a
    plain `pip install` outright with "externally-managed-environment", independent of whether
    the package or network is fine; harmless to always pass since pip ignores it on interpreters
    that aren't externally managed (a plain venv, Windows, etc.). (2) a plain `pip install torch`
    resolves to the CUDA build by default, silently pulling several GB of nvidia-* wheels
    (cublas/cudnn/nccl/...) that are dead weight here -- the contract in prompt_templates.py is
    explicit that training must fit the wall-clock budget on CPU only, no GPU available -- so
    `torch` specifically is pinned to PyPI's CPU-only wheel index; `torchfm` (pure Python, depends
    on torch) still resolves from the default index and picks up the already-installed CPU torch
    without re-resolving a different build. timeout bumped from 300s->600s since even the CPU-only
    torch wheel alone is ~120MB and an unattended run shouldn't spuriously fail on a slower link."""
    cmd = [sys.executable, '-m', 'pip', 'install', package, '--break-system-packages']
    if package == 'torch':
        cmd += ['--index-url', _TORCH_CPU_INDEX]
    returncode, output, timed_out = probe_runner.stream_subprocess(cmd, timeout)
    if timed_out:
        return False, f'pip install {package} exceeded {timeout}s'
    return returncode == 0, output


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

    installed_packages = set()  # v4 roadmap Phase 3b -- never install the same allowlisted
                                  # package twice in one node; bounds self-heal to at most
                                  # len(PIP_INSTALL_ALLOWLIST) extra subprocess calls ever
    repairs_left = args.max_repairs
    while not result['ok']:
        missing = _detect_missing_package(result['error'])
        if missing and missing in PIP_INSTALL_ALLOWLIST and missing not in installed_packages:
            installed_packages.add(missing)
            display.phase(f'node {node_id}: detected missing allowlisted package {missing!r} in '
                          f'the traceback -- installing into {sys.executable} and retrying the '
                          f'same code unchanged (no LLM call, doesn\'t consume --max_repairs)...')
            ok, output = pip_install(missing)
            run_attempts.append({'attempt': len(run_attempts), 'action': 'pip_install',
                                  'package': missing, 'ok': ok, 'output': output[-2000:]})
            if ok:
                display.run_start(f'candidate (node {node_id}, after installing {missing})', timeout)
                t0 = time.time()
                result = run_candidate(iter_dir, args.data_dir, metrics_path, hparams, args.seed, timeout,
                                        verbose=not args.quiet)
                display.run_end(t0, result['ok'], timed_out=result.get('timed_out', False))
                run_attempts.append({'attempt': len(run_attempts), 'ok': result['ok'], 'error': result.get('error')})
                continue  # re-check result['ok'] above -- doesn't consume repairs_left
            # install itself failed (e.g. no network) -- fall through to the normal LLM-repair
            # path below rather than giving up outright, same as any other unresolved failure;
            # the LLM still gets a chance to rewrite around the missing dependency as usual.

        if repairs_left <= 0:
            break
        repairs_left -= 1
        repair_prompt = pt.build_repair_prompt(read_code(iter_dir), hypothesis, result['error'])
        attempt, tries = probe_runner.call_llm_with_retry(
            pt.build_system_prompt(args.torch_available), repair_prompt, args.model, args.max_budget_usd,
            label=f'repairing candidate after run failure ({repairs_left + 1} attempt(s) left)',
            timeout=args.propose_timeout)
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


def run_parallel_variants(node_id, best_dir, variants, args):
    """Runs every variant from a `parallel_hypothesis` propose turn concurrently (each
    its own subprocess -- run_candidate() already shells out via stream_subprocess, so
    running several from different threads is just several independent OS processes,
    no shared interpreter state to worry about). Deliberately NO repair-retry per
    variant (unlike run_and_repair) -- with several variants in flight, retrying every
    failing one would multiply LLM cost by the variant count for what's supposed to be
    a cheap way to explore several ideas at once; a variant that fails just loses the
    race. Per-epoch live streaming is suppressed (verbose=False) since interleaving
    several variants' epoch lines on one terminal would be illegible -- a one-line
    summary per variant is printed as each finishes instead.

    Returns (variant_dirs, results, best_index, best_result) where variant_dirs[i] is
    the candidate directory for variants[i], results[i] is that variant's
    run_candidate()-shaped dict, and best_index/best_result identify whichever variant
    had the highest valid primary among the ones that ran successfully (best_index is
    None if every variant failed)."""
    variant_dirs = []
    for i in range(len(variants)):
        vdir = RUNS_DIR / f'node_{node_id}_v{i + 1}'
        if vdir.exists():
            shutil.rmtree(vdir)
        vdir.mkdir(parents=True)
        for fname in pt.ALLOWED_FILES:
            shutil.copy2(best_dir / fname, vdir / fname)
        apply_files(vdir, variants[i]['files'])
        variant_dirs.append(vdir)

    hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs}
    results = [None] * len(variants)

    def _run_one(i):
        metrics_path = variant_dirs[i] / 'metrics.json'
        t0 = time.time()
        r = run_candidate(variant_dirs[i], args.data_dir, metrics_path, hparams, args.seed,
                           args.run_timeout, verbose=False)
        r['elapsed_s'] = time.time() - t0
        return i, r

    display.phase(f'running {len(variants)} variants in parallel (per-epoch output suppressed '
                  f'for legibility -- see logs/node_{node_id}.json for full detail after)...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(variants)) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(_run_one, i) for i in range(len(variants))]):
            i, r = fut.result()
            results[i] = r
            if r['ok']:
                display.phase(f'  variant {i + 1}: ok in {r["elapsed_s"]:.0f}s -- '
                              f'valid primary {r["metrics"]["valid"]["primary"]:.4f}')
            else:
                display.phase(f'  variant {i + 1}: FAILED in {r["elapsed_s"]:.0f}s -- '
                              f'{(r.get("error") or "")[:120]}')

    best_i, best_result = None, None
    for i, r in enumerate(results):
        if r['ok'] and (best_result is None or r['metrics']['valid']['primary'] > best_result['metrics']['valid']['primary']):
            best_i, best_result = i, r
    return variant_dirs, results, best_i, best_result


def run_variant_sweep(node_id, variant_dir, args, sweep_param, sweep_values, base_result):
    """Hyperparameter sweep applied ONLY to whichever variant already won the initial
    parallel race, per the user's framing: sweeping is a refinement of an already-chosen
    mechanism, not a way to choose between mechanisms, so there's no point spending it on
    variants that already lost. `base_result` is that variant's already-computed
    run_candidate()-shaped result at plain default hyperparameters (from the initial
    race) -- counted as one trial alongside the swept values, not re-run. No
    repair-retry for any sweep value, unlike run_sweep()'s first trial: this code is
    already proven to run (it's how it won the race in the first place), so a failing
    sweep value is a bad hyperparameter, not a bug worth an LLM repair call.
    Returns (best_result, best_value, trials) -- trials includes a leading 'default'
    entry for base_result plus one entry per swept value."""
    base_hparams = {'k': args.k, 'lr': args.lr, 'epochs': args.epochs}
    metrics_path = variant_dir / 'metrics.json'
    trials = [{'value': 'default', 'ok': True, 'primary': base_result['metrics']['valid']['primary'],
               'error': None}]
    best_result, best_value = base_result, 'default'
    for value in sweep_values:
        display.run_start(f'candidate (node {node_id}, winning variant, {sweep_param}={value})',
                           args.run_timeout)
        t0 = time.time()
        r = run_candidate(variant_dir, args.data_dir, metrics_path, {**base_hparams, sweep_param: value},
                           args.seed, args.run_timeout, verbose=not args.quiet)
        display.run_end(t0, r['ok'], timed_out=r.get('timed_out', False))
        trials.append({'value': value, 'ok': r['ok'],
                        'primary': r['metrics']['valid']['primary'] if r['ok'] else None,
                        'error': None if r['ok'] else r.get('error')})
        if r['ok'] and r['metrics']['valid']['primary'] > best_result['metrics']['valid']['primary']:
            best_result, best_value = r, value
    return best_result, best_value, trials


def append_probe_finding(node_id, question, result):
    PROBE_FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    block = f"### Probe (node {node_id})\nQuestion: {question}\n\n```json\n{json.dumps(result, indent=2)}\n```\n\n"
    with open(PROBE_FINDINGS_PATH, 'a', encoding='utf-8') as f:
        f.write(block)


def current_best_lever_category(history):
    """The lever_category of whichever node is currently accepted, or None if that
    node predates LEVER_CATEGORY tracking (true for the earliest accepts in this
    project) or nothing has been accepted yet."""
    for h in reversed(history):
        if h.get('status') == 'accepted':
            return h.get('lever_category')
    return None


def best_alternate_candidate_code(history):
    """Full code of a past candidate that ISN'T the current accepted best, so an
    'ensembling' hypothesis can reuse it verbatim instead of re-deriving it from a text
    description in history -- the propose LLM has no filesystem access of its own
    (--tools ""), so without this it would have to reimplement a past mechanism from
    memory, reintroducing exactly the bug risk ensembling is supposed to avoid by
    reusing already-validated code.

    Prefers the highest-scoring alternate whose lever_category DIFFERS from the current
    best's -- node 17 ensembled two candidates from the SAME category (plain BPR-FM +
    the same FM's auxiliary-head variant) and saw essentially no gain (0.6039 either
    way), because near-identical mechanisms make near-identical errors; diversity of
    mechanism, not just a high individual score, is what makes averaging actually
    reduce variance. Falls back to the plain highest-scoring alternate if the current
    best's category is unknown (true for nodes accepted before this tracking existed)
    or no differently-categorized alternate has code on disk.

    Excludes past 'ensembling' attempts from the candidate pool entirely -- once node 17
    existed and outscored node 14 individually, a naive highest-score fallback started
    suggesting "reuse node 17's code" as the ensembling partner, which is circular
    (offering an already-combined candidate as the fresh single mechanism to combine
    with next) rather than useful.

    Returns (node_id, code_dict) or (None, None) if no candidate exists on disk at all
    (e.g. the top scorer IS the current best, or its node_N/ dir was cleaned up).
    Bounded to at most one extra file pair regardless of run length -- doesn't
    reintroduce the unbounded per-node growth the history-windowing fix addressed."""
    best_category = current_best_lever_category(history)
    alternates = [h for h in pt.top_candidates(history, top_k=10)
                  if h.get('status') != 'accepted' and h.get('lever_category') != 'ensembling']

    def _code_if_on_disk(h):
        node_dir = RUNS_DIR / f"node_{h['iter']}"
        if (node_dir / 'data.py').exists() and (node_dir / 'baseline.py').exists():
            return h['iter'], read_code(node_dir)
        return None

    if best_category:
        for h in alternates:
            if h.get('lever_category') and h['lever_category'] != best_category:
                found = _code_if_on_disk(h)
                if found:
                    return found

    for h in alternates:
        found = _code_if_on_disk(h)
        if found:
            return found
    return None, None


def _finalize_hypothesis_node(node_id, iter_dir, hypothesis, expected_effect, lever_category,
                               reflection, changed, best_dir, best_primary, prev_metrics, args,
                               llm_calls, sweep_param=None, sweep_values=None, extra_record_fields=None):
    """Shared tail of run_iteration()'s plain hypothesis-mode branch AND its code_session
    branch (v4 roadmap Phase 4) -- runs the candidate already sitting in `iter_dir` (with an
    optional hyperparameter sweep), accepts it into `best_dir` on improvement, writes the
    log record, and returns the (status, new_best_primary, new_best_metrics, history_entry)
    tuple run_iteration() itself returns. `changed` is the already-known list of touched
    ALLOWED_FILES (computed differently by each caller -- from the parsed files dict for a
    plain hypothesis turn, from a disk diff against the pre-session snapshot for a code
    session -- but identical in meaning from here on). `extra_record_fields` (e.g.
    {'authored_via': 'code_session', 'code_session_question': ...}) is merged into both the
    on-disk log record and the returned history entry; None for the plain hypothesis path."""
    extra = extra_record_fields or {}
    sweep = None
    if sweep_param:
        result, best_value, trials, run_attempts = run_sweep(node_id, iter_dir, hypothesis, args, llm_calls,
                                                               sweep_param, sweep_values)
        sweep = {'param': sweep_param, 'trials': trials, 'best_value': best_value}
    else:
        result, run_attempts = run_and_repair(node_id, iter_dir, hypothesis, args, llm_calls)

    if not result['ok']:
        record = {'iter': node_id, 'status': 'failed', 'hypothesis': hypothesis,
                   'reflection': reflection, 'expected_effect': expected_effect,
                   'lever_category': lever_category, 'changed_files': changed,
                   'sweep': sweep, 'error_summary': result['error'], 'llm_calls': llm_calls,
                   'run_attempts': run_attempts, **extra}
        write_log(node_id, record)
        return 'failed', best_primary, prev_metrics, {'iter': node_id, 'status': 'failed',
                                         'hypothesis': hypothesis, 'error_summary': result['error'], **extra}

    new_primary = result['metrics']['valid']['primary']
    if new_primary > best_primary + args.converge_eps:
        for fname in pt.ALLOWED_FILES:
            shutil.copy2(iter_dir / fname, best_dir / fname)
        status, out_best, out_metrics = 'accepted', new_primary, result['metrics']
    else:
        status, out_best, out_metrics = 'rejected', best_primary, prev_metrics

    record = {'iter': node_id, 'status': status, 'hypothesis': hypothesis,
              'reflection': reflection, 'expected_effect': expected_effect,
              'lever_category': lever_category, 'changed_files': changed,
              'sweep': sweep, 'metrics': result['metrics'], 'prev_metrics': prev_metrics,
              'training_curve': result.get('training_curve'),
              'prev_best_primary': best_primary, 'new_primary': new_primary,
              'llm_calls': llm_calls, 'run_attempts': run_attempts, **extra}
    write_log(node_id, record)
    return status, out_best, out_metrics, {'iter': node_id, 'status': status, 'hypothesis': hypothesis,
                               'reflection': reflection, 'expected_effect': expected_effect,
                               'lever_category': lever_category, 'sweep': sweep,
                               'metrics': result['metrics'], 'prev_metrics': prev_metrics,
                               'training_curve': result.get('training_curve'),
                               'primary': new_primary, 'prev_best': best_primary, **extra}


def run_iteration(node_id, best_dir, best_primary, prev_metrics, history, args, streak,
                   web_research_remaining, explore_remaining, eda_round_remaining,
                   code_session_remaining):
    """One node of the single iteration chain. Proposes; the LLM itself picks a
    hypothesis+code-change turn, a probe turn, a web-research turn, a repo-explore
    turn, an EDA-round turn, or a code-session turn (budget permitting for the last
    four). Accepts a hypothesis into `best_dir` (in place) on improvement. Returns
    (status, new_best_primary, new_best_metrics, history_entry). status is one of
    accepted/rejected/answered/failed/researched/research_failed/research_denied/
    explored/explore_failed/explore_denied/eda_round/eda_round_failed/eda_round_denied/
    code_session_denied (a code session's ATTEMPTED outcome reuses accepted/rejected/
    failed rather than its own status family -- see _finalize_hypothesis_node -- marked
    instead via the returned entry's 'authored_via': 'code_session' field)."""
    best_code = read_code(best_dir)
    alt_node_id, alt_code = best_alternate_candidate_code(history)
    propose_prompt = pt.build_propose_prompt(best_code, history, best_primary, streak,
                                              args.escalate_after, args.converge_eps,
                                              web_research_remaining=web_research_remaining,
                                              explore_remaining=explore_remaining,
                                              eda_round_remaining=eda_round_remaining,
                                              code_session_remaining=code_session_remaining,
                                              alt_node_id=alt_node_id, alt_code=alt_code)
    llm_calls = []

    attempt, tries = probe_runner.call_llm_with_retry(
        pt.build_system_prompt(args.torch_available), propose_prompt, args.model, args.max_budget_usd,
        label='deciding the next move (hypothesis or probe)', timeout=args.propose_timeout)
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

        result, run_attempts = probe_runner.run_probe_and_repair(
            probe_dir, question, args.data_dir, args.model, args.max_budget_usd,
            args.propose_timeout, args.probe_timeout, args.max_repairs, llm_calls,
            system_prompt=pt.build_system_prompt(args.torch_available), allowed_files=pt.PROBE_ALLOWED_FILES)
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

    if parsed['mode'] == 'research':
        question = parsed['research_question']
        display.field('research question', question)

        if web_research_remaining <= 0:
            record = {'iter': node_id, 'status': 'research_denied', 'research_question': question,
                       'reflection': reflection, 'llm_calls': llm_calls}
            write_log(node_id, record)
            return ('research_denied', best_primary, prev_metrics,
                    {'iter': node_id, 'status': 'research_denied', 'research_question': question,
                     'reflection': reflection})

        result = web_research.run_research(question, node_id, model=args.model,
                                            max_budget_usd=args.max_budget_usd,
                                            timeout=args.web_research_timeout)
        llm_calls.append({'cost_usd': result['cost_usd'], 'ok': result['ok'],
                           'error': None if result['ok'] else result['rejected_reason']})

        if result['ok']:
            display.field('research result', f"found: {result['note']['title']} ({result['note']['source_url']})")
            rag.run()  # regenerate literature_context.md so the new note is visible next propose call
            record = {'iter': node_id, 'status': 'researched', 'research_question': question,
                       'reflection': reflection, 'note': result['note'], 'saved_path': result['saved_path'],
                       'llm_calls': llm_calls}
            write_log(node_id, record)
            return ('researched', best_primary, prev_metrics,
                    {'iter': node_id, 'status': 'researched', 'research_question': question,
                     'reflection': reflection, 'note_title': result['note']['title']})

        display.field('research result', f"none found ({result['rejected_reason']})")
        record = {'iter': node_id, 'status': 'research_failed', 'research_question': question,
                   'reflection': reflection, 'rejected_reason': result['rejected_reason'],
                   'llm_calls': llm_calls}
        write_log(node_id, record)
        return ('research_failed', best_primary, prev_metrics,
                {'iter': node_id, 'status': 'research_failed', 'research_question': question,
                 'reflection': reflection})

    if parsed['mode'] == 'explore':
        question = parsed['explore_question']
        display.field('explore question', question)

        if explore_remaining <= 0:
            record = {'iter': node_id, 'status': 'explore_denied', 'explore_question': question,
                       'reflection': reflection, 'llm_calls': llm_calls}
            write_log(node_id, record)
            return ('explore_denied', best_primary, prev_metrics,
                    {'iter': node_id, 'status': 'explore_denied', 'explore_question': question,
                     'reflection': reflection})

        result = repo_explore.run_explore(question, node_id, model=args.model,
                                           max_budget_usd=args.max_budget_usd,
                                           timeout=args.explore_timeout)
        llm_calls.append({'cost_usd': result['cost_usd'], 'ok': result['ok'],
                           'error': None if result['ok'] else result['rejected_reason']})

        if result['ok']:
            display.field('explore result', f"found: {result['note']['title']} ({result['note']['citation']})")
            rag.run()  # regenerate literature_context.md so the new note is visible next propose call
            record = {'iter': node_id, 'status': 'explored', 'explore_question': question,
                       'reflection': reflection, 'note': result['note'], 'saved_path': result['saved_path'],
                       'llm_calls': llm_calls}
            write_log(node_id, record)
            return ('explored', best_primary, prev_metrics,
                    {'iter': node_id, 'status': 'explored', 'explore_question': question,
                     'reflection': reflection, 'note_title': result['note']['title']})

        display.field('explore result', f"none found ({result['rejected_reason']})")
        record = {'iter': node_id, 'status': 'explore_failed', 'explore_question': question,
                   'reflection': reflection, 'rejected_reason': result['rejected_reason'],
                   'llm_calls': llm_calls}
        write_log(node_id, record)
        return ('explore_failed', best_primary, prev_metrics,
                {'iter': node_id, 'status': 'explore_failed', 'explore_question': question,
                 'reflection': reflection})

    if parsed['mode'] == 'eda_round':
        question = parsed['eda_round_question']
        display.field('eda round request', question)

        if eda_round_remaining <= 0:
            record = {'iter': node_id, 'status': 'eda_round_denied', 'eda_round_question': question,
                       'reflection': reflection, 'llm_calls': llm_calls}
            write_log(node_id, record)
            return ('eda_round_denied', best_primary, prev_metrics,
                    {'iter': node_id, 'status': 'eda_round_denied', 'eda_round_question': question,
                     'reflection': reflection})

        report = json.loads(eda.REPORT_PATH.read_text(encoding='utf-8'))
        round_result = eda_agent.run_agentic_exploration(
            report, args.data_dir, model=args.model, max_turns=args.eda_agent_turns,
            max_budget_usd=args.eda_agent_max_budget_usd, max_repairs=args.max_repairs,
            propose_timeout=args.propose_timeout, probe_timeout=args.probe_timeout,
            out_dir=eda.RUNS_DIR, focus=question, context_note=reflection)
        llm_calls.extend(c for t in round_result['new_turns'] for c in t.get('llm_calls', []))
        run_attempts = [ra for t in round_result['new_turns'] for ra in t.get('run_attempts', [])]
        n_new_probes = sum(1 for t in round_result['new_turns'] if t['mode'] == 'probe')

        if n_new_probes > 0:
            new_summary = eda.summarize_with_llm(
                report, args.model, args.max_budget_usd,
                extra_context=eda_agent.accumulated_findings_text(eda.RUNS_DIR))
            eda.SUMMARY_PATH.write_text(new_summary, encoding='utf-8')
            display.field('eda round result', f'{n_new_probes} new finding(s), eda_summary.md refreshed')
            record = {'iter': node_id, 'status': 'eda_round', 'eda_round_question': question,
                       'reflection': reflection, 'eda_round': round_result['round'],
                       'stopped_reason': round_result['stopped_reason'],
                       'llm_calls': llm_calls, 'run_attempts': run_attempts}
            write_log(node_id, record)
            return ('eda_round', best_primary, prev_metrics,
                    {'iter': node_id, 'status': 'eda_round', 'eda_round_question': question,
                     'reflection': reflection, 'n_new_probes': n_new_probes})

        display.field('eda round result', f"no new findings ({round_result['stopped_reason']})")
        record = {'iter': node_id, 'status': 'eda_round_failed', 'eda_round_question': question,
                   'reflection': reflection, 'eda_round': round_result['round'],
                   'stopped_reason': round_result['stopped_reason'],
                   'llm_calls': llm_calls, 'run_attempts': run_attempts}
        write_log(node_id, record)
        return ('eda_round_failed', best_primary, prev_metrics,
                {'iter': node_id, 'status': 'eda_round_failed', 'eda_round_question': question,
                 'reflection': reflection})

    if parsed['mode'] == 'code_session':
        question = parsed['code_session_question']
        display.field('code session request', question)

        if code_session_remaining <= 0:
            record = {'iter': node_id, 'status': 'code_session_denied', 'code_session_question': question,
                       'reflection': reflection, 'llm_calls': llm_calls}
            write_log(node_id, record)
            return ('code_session_denied', best_primary, prev_metrics,
                    {'iter': node_id, 'status': 'code_session_denied', 'code_session_question': question,
                     'reflection': reflection})

        iter_dir = snapshot_node_dir(node_id, best_dir)  # pre-populates data.py/baseline.py with
                                                            # the current best -- best_code above is
                                                            # this same content, used below to diff
        session_result = code_agent.run_code_session(
            iter_dir, best_code, history, question, model=args.model,
            max_budget_usd=args.code_session_max_budget_usd, timeout=args.code_session_timeout,
            torch_available=args.torch_available)
        llm_calls.append({'cost_usd': session_result.get('cost_usd'), 'ok': session_result['ok'],
                           'error': session_result.get('error')})

        if not session_result['ok']:
            error_summary = f"code session call failed: {session_result['error']}"
            record = {'iter': node_id, 'status': 'failed', 'hypothesis': f'(code session) {question}',
                       'reflection': reflection, 'error_summary': error_summary, 'llm_calls': llm_calls,
                       'run_attempts': [], 'authored_via': 'code_session', 'code_session_question': question}
            write_log(node_id, record)
            return 'failed', best_primary, prev_metrics, {'iter': node_id, 'status': 'failed',
                                             'hypothesis': f'(code session) {question}',
                                             'error_summary': error_summary,
                                             'authored_via': 'code_session'}

        changed = [f for f in pt.ALLOWED_FILES
                   if (iter_dir / f).read_text(encoding='utf-8') != best_code[f]]
        if not changed:
            record = {'iter': node_id, 'status': 'failed', 'hypothesis': f'(code session) {question}',
                       'reflection': reflection, 'error_summary': 'code session made no changes to data.py/baseline.py',
                       'raw_response': session_result['text'][:2000], 'llm_calls': llm_calls, 'run_attempts': [],
                       'authored_via': 'code_session', 'code_session_question': question}
            write_log(node_id, record)
            return 'failed', best_primary, prev_metrics, {'iter': node_id, 'status': 'failed',
                                             'hypothesis': f'(code session) {question}',
                                             'error_summary': 'code session made no changes',
                                             'authored_via': 'code_session'}

        session_parsed = pt.parse_response(session_result['text'])
        cs_hypothesis = session_parsed['hypothesis'] or f'(code session) {question}'
        cs_expected_effect = session_parsed['expected_effect']
        cs_lever_category = session_parsed['lever_category']
        display.field('hypothesis', cs_hypothesis)
        display.field('expected effect', cs_expected_effect)
        display.field('lever category', cs_lever_category)
        return _finalize_hypothesis_node(
            node_id, iter_dir, cs_hypothesis, cs_expected_effect, cs_lever_category, reflection,
            changed, best_dir, best_primary, prev_metrics, args, llm_calls,
            extra_record_fields={'authored_via': 'code_session', 'code_session_question': question})

    if parsed['mode'] == 'parallel_hypothesis':
        variants = parsed['variants']
        display.field('parallel variants', '; '.join(
            f"[{i + 1}] ({v['lever_category']}) {v['hypothesis'][:60]}" for i, v in enumerate(variants)))

        variant_dirs, results, best_i, best_result = run_parallel_variants(node_id, best_dir, variants, args)

        variant_records = [
            {'variant': i + 1, 'hypothesis': v['hypothesis'], 'expected_effect': v['expected_effect'],
             'lever_category': v['lever_category'], 'ok': r['ok'],
             'metrics': r.get('metrics') if r['ok'] else None,
             'error': None if r['ok'] else r.get('error')}
            for i, (v, r) in enumerate(zip(variants, results))
        ]

        if best_result is None:
            hyp_summary = f'{len(variants)} parallel variants, all failed'
            record = {'iter': node_id, 'status': 'failed', 'hypothesis': hyp_summary,
                       'reflection': reflection, 'variants': variant_records,
                       'error_summary': 'every parallel variant failed to run',
                       'llm_calls': llm_calls, 'run_attempts': []}
            write_log(node_id, record)
            return ('failed', best_primary, prev_metrics,
                    {'iter': node_id, 'status': 'failed', 'hypothesis': hyp_summary,
                     'error_summary': 'every parallel variant failed to run'})

        winning_hypothesis = variants[best_i]['hypothesis']
        winning_expected_effect = variants[best_i]['expected_effect']
        winning_lever_category = variants[best_i]['lever_category']
        sweep = None
        win_sweep_param, win_sweep_values = variants[best_i]['sweep_param'], variants[best_i]['sweep_values']
        if win_sweep_param:
            display.field('sweeping winner', f'v{best_i + 1} over {win_sweep_param}={win_sweep_values}')
            best_result, best_value, trials = run_variant_sweep(
                node_id, variant_dirs[best_i], args, win_sweep_param, win_sweep_values, best_result)
            sweep = {'param': win_sweep_param, 'trials': trials, 'best_value': best_value}

        new_primary = best_result['metrics']['valid']['primary']
        if new_primary > best_primary + args.converge_eps:
            for fname in pt.ALLOWED_FILES:
                shutil.copy2(variant_dirs[best_i] / fname, best_dir / fname)
            status, out_best, out_metrics = 'accepted', new_primary, best_result['metrics']
        else:
            status, out_best, out_metrics = 'rejected', best_primary, prev_metrics

        record = {'iter': node_id, 'status': status, 'hypothesis': winning_hypothesis,
                   'reflection': reflection, 'expected_effect': winning_expected_effect,
                   'lever_category': winning_lever_category,
                   'changed_files': list(variants[best_i]['files'].keys()), 'variants': variant_records,
                   'winning_variant': best_i + 1, 'sweep': sweep, 'metrics': best_result['metrics'],
                   'prev_metrics': prev_metrics, 'training_curve': best_result.get('training_curve'),
                   'prev_best_primary': best_primary, 'new_primary': new_primary,
                   'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return (status, out_best, out_metrics,
                {'iter': node_id, 'status': status, 'hypothesis': winning_hypothesis, 'reflection': reflection,
                 'expected_effect': winning_expected_effect, 'lever_category': winning_lever_category,
                 'sweep': sweep, 'metrics': best_result['metrics'],
                 'prev_metrics': prev_metrics, 'training_curve': best_result.get('training_curve'),
                 'primary': new_primary, 'prev_best': best_primary, 'variants': variant_records,
                 'winning_variant': best_i + 1})

    # hypothesis mode
    hypothesis, expected_effect, files = parsed['hypothesis'], parsed['expected_effect'], parsed['files']
    lever_category = parsed['lever_category']
    sweep_param, sweep_values = parsed['sweep_param'], parsed['sweep_values']
    display.field('hypothesis', hypothesis)
    display.field('expected effect', expected_effect)
    display.field('lever category', lever_category)
    if sweep_param:
        display.field('sweep', f'{sweep_param} over {sweep_values}')
    changed = [f for f in files if f in pt.ALLOWED_FILES]
    if not changed:
        record = {'iter': node_id, 'status': 'failed', 'hypothesis': hypothesis,
                   'reflection': reflection, 'expected_effect': expected_effect,
                   'lever_category': lever_category,
                   'error_summary': 'no valid file changes parsed from LLM response',
                   'raw_response': attempt['text'][:2000], 'llm_calls': llm_calls, 'run_attempts': []}
        write_log(node_id, record)
        return 'failed', best_primary, prev_metrics, {'iter': node_id, 'status': 'failed',
                                         'hypothesis': hypothesis, 'error_summary': 'no file changes proposed'}

    iter_dir = snapshot_node_dir(node_id, best_dir)
    apply_files(iter_dir, files)
    return _finalize_hypothesis_node(node_id, iter_dir, hypothesis, expected_effect, lever_category,
                                      reflection, changed, best_dir, best_primary, prev_metrics, args,
                                      llm_calls, sweep_param=sweep_param, sweep_values=sweep_values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(WORKSPACE / 'data'))
    ap.add_argument('--iterations', type=int, default=10, help='max LLM-proposed iterations')
    ap.add_argument('--max_repairs', type=int, default=2, help='error-repair attempts per node')
    ap.add_argument('--run_timeout', type=int, default=400,
                     help='seconds before any candidate run is killed (covers both numpy and torch candidates)')
    ap.add_argument('--propose_timeout', type=int, default=300,
                     help='seconds before a propose/repair LLM call is killed (was hardcoded to '
                          'llm.py\'s 180s default with no override -- raised and exposed here because '
                          'the propose prompt grows with the run (more history, more literature notes, '
                          'the lever-category taxonomy), so a timeout fine at node 1 can start failing '
                          'purely from prompt size by node 10+, not any actual problem with the call)')
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
    ap.add_argument('--web_research_budget', type=int, default=2,
                     help='max live web-research calls for this run (separate pool from '
                          '--iterations, like the old --diagnosis_budget) -- 0 disables the '
                          'RESEARCH_QUESTION option entirely')
    ap.add_argument('--web_research_timeout', type=int, default=120,
                     help='seconds before a web-research LLM call is killed')
    ap.add_argument('--explore_budget', type=int, default=2,
                     help='max repo-explore calls for this run (separate pool from --iterations '
                          'and from --web_research_budget) -- 0 disables the EXPLORE_QUESTION '
                          'option entirely')
    ap.add_argument('--explore_timeout', type=int, default=90,
                     help='seconds before a repo-explore LLM call is killed (read-only local '
                          'Read/Grep/Glob, so shorter than the web-research default)')
    ap.add_argument('--eda_round_budget', type=int, default=2,
                     help='max mid-run agentic-EDA rounds the LLM may request for this run '
                          '(separate pool from --iterations and the other budgets above) -- '
                          '0 disables the EDA_ROUND_REQUEST option entirely')
    ap.add_argument('--code_session_budget', type=int, default=2,
                     help='max bounded multi-turn coding sessions (v4 roadmap Phase 4) the LLM '
                          'may request for this run (separate pool from --iterations and the '
                          'other budgets above) -- 0 disables the CODE_SESSION_REQUEST option '
                          'entirely')
    ap.add_argument('--code_session_max_budget_usd', type=float, default=1.00,
                     help='cumulative $ ceiling for ONE coding session (all its internal tool-use '
                          'turns combined, enforced by the claude CLI itself) -- separate from '
                          '--max_budget_usd, which only caps a single plain propose/repair call')
    ap.add_argument('--code_session_timeout', type=int, default=480,
                     help='seconds before a coding session subprocess is killed -- longer than '
                          '--propose_timeout since a multi-turn read/edit/verify session needs '
                          'more wall-clock than one text completion')
    ap.add_argument('--eda_agent_turns', type=int, default=3,
                     help='max write-probe -> run -> see-result turns per agentic-EDA round '
                          '(both the mandatory startup pass and every mid-run round) before '
                          'finalizing; 0 disables the loop entirely, falling back to exactly '
                          'the single pinned summarization call from before Phase 2')
    ap.add_argument('--eda_agent_max_budget_usd', type=float, default=1.00,
                     help='cumulative $ ceiling across one whole agentic-EDA round (all turns '
                          '+ repairs combined) -- separate from --max_budget_usd, which only '
                          'caps a single LLM call')
    ap.add_argument('--skip_eda_agent', action='store_true',
                     help='disable the agentic-EDA loop entirely -- both the mandatory startup '
                          'pass (same as --eda_agent_turns 0) and the mid-run EDA_ROUND_REQUEST '
                          'option (same as --eda_round_budget 0)')
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

    preflight_check(args)
    ensure_eda(args)
    ensure_literature(args)
    if args.resume:
        best_dir = ensure_best_dir()
    else:
        logs_backup, runs_backup = archive_prior_run()
        if logs_backup or runs_backup:
            display.phase(f'--no-resume: backed up previous logs to {logs_backup}, '
                           f'previous agent/runs state to {runs_backup}')
        best_dir = reset_best_dir()
        clear_prior_logs()  # defensive no-op in the common case -- archive_prior_run() already
                             # moved every logs/node_*.json (N>=1) out, this just catches leftovers

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
    write_log(0, {'iter': 0, 'status': label, 'metrics': result['metrics'],
                  'preflight': {'torch_available': args.torch_available, 'interpreter': sys.executable}})
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
    web_research_remaining = args.web_research_budget  # a per-invocation budget, like plateau_streak,
                                                         # not reconstructed from prior sessions' spend
    explore_remaining = args.explore_budget  # same per-invocation-only accounting as web_research_remaining
    eda_round_remaining = 0 if args.skip_eda_agent else args.eda_round_budget
    code_session_remaining = args.code_session_budget  # same per-invocation-only accounting
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
            node_id, best_dir, best_primary, best_metrics, history, args, plateau_streak,
            web_research_remaining, explore_remaining, eda_round_remaining, code_session_remaining)
        history.append(entry)

        if status in ('researched', 'research_failed'):
            web_research_remaining -= 1  # a denied request never spent a research call
        if status in ('explored', 'explore_failed'):
            explore_remaining -= 1  # a denied request never spent an explore call
        if status in ('eda_round', 'eda_round_failed'):
            eda_round_remaining -= 1  # a denied request never spent an eda-round
        if entry.get('authored_via') == 'code_session' and status != 'code_session_denied':
            # a code session's ATTEMPTED outcome reuses accepted/rejected/failed (see
            # _finalize_hypothesis_node), unlike research/explore/eda_round whose own status
            # strings already say "attempted" -- so the decrement check has to key off the
            # 'authored_via' marker instead of the status string
            code_session_remaining -= 1

        if status in ('accepted', 'rejected'):
            display.result_line(entry['status'], entry.get('hypothesis'), entry.get('primary'),
                                 entry.get('prev_best'), entry.get('error_summary'))
        elif status == 'answered':
            display.probe_line('answered', entry.get('question'))
        elif status in ('researched', 'research_failed', 'research_denied'):
            display.research_line(status, entry.get('research_question'), entry.get('note_title'))
        elif status in ('explored', 'explore_failed', 'explore_denied'):
            display.explore_line(status, entry.get('explore_question'), entry.get('note_title'))
        elif status in ('eda_round', 'eda_round_failed', 'eda_round_denied'):
            display.eda_round_line(status, entry.get('eda_round_question'), entry.get('n_new_probes'))
        elif status == 'code_session_denied':
            display.code_session_line(status, entry.get('code_session_question'))
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
