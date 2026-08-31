"""Shared, LLM-role-agnostic execution primitives: streaming a subprocess with a
wall-clock timeout, running a probe.py through eda_probe.py's fixed harness, retrying a
zero-tool LLM call, and the retry-with-traceback-feedback repair loop for a probe.

Deliberately a leaf module (imports only display/llm/prompt_templates + stdlib, never
eda/orchestrator/rag/repo_explore/web_research) so both orchestrator.py (today's mid-run
PROBE_QUESTION path) and eda_agent.py (the agentic-EDA exploration loop, called both at
startup and mid-run) can import it without a load-order-dependent cycle. Previously this
logic lived inline in orchestrator.py, hardcoded to prompt_templates.SYSTEM_PROMPT/
PROBE_ALLOWED_FILES; run_probe_and_repair() now takes those as explicit params so one
implementation serves both roles instead of being duplicated.
"""
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import display
import llm
import prompt_templates as pt

EDA_PROBE = Path(__file__).resolve().parent / 'eda_probe.py'


def _pump_stream(pipe, q):
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


def run_probe_candidate(probe_dir, data_dir, out_path, timeout):
    """Runs a probe.py through agent/eda_probe.py's fixed harness. Same success/failure
    contract as orchestrator.py's run_candidate: exit 0 + valid JSON at --out = success."""
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


def call_llm_with_retry(system_prompt, user_prompt, model, max_budget_usd, label, retries=1,
                         timeout=llm.DEFAULT_TIMEOUT_S):
    """timeout defaults to llm.py's fixed 180s if not passed explicitly -- but every real
    call site should pass its own role-appropriate timeout, since a prompt that grows
    over a run (more history, more literature notes) can start timing out later even
    though a fixed default was fine early on."""
    t0 = display.llm_call_start(label)
    attempt = llm.call_claude(system_prompt, user_prompt, model=model, max_budget_usd=max_budget_usd,
                               timeout=timeout)
    display.llm_call_end(t0, attempt)
    tries = [attempt]
    while not attempt['ok'] and retries > 0:
        retries -= 1
        display.retrying()
        t0 = display.llm_call_start(label)
        attempt = llm.call_claude(system_prompt, user_prompt, model=model, max_budget_usd=max_budget_usd,
                                   timeout=timeout)
        display.llm_call_end(t0, attempt)
        tries.append(attempt)
    return attempt, tries


def run_probe_and_repair(probe_dir, question, data_dir, model, max_budget_usd,
                          propose_timeout, probe_timeout, max_repairs, llm_calls,
                          system_prompt, allowed_files):
    """Same retry-with-traceback-feedback pattern as orchestrator.py's run_and_repair,
    pointed at a probe.py instead of data.py/baseline.py. system_prompt and
    allowed_files are explicit params (not hardcoded to prompt_templates.SYSTEM_PROMPT/
    PROBE_ALLOWED_FILES) so this one implementation serves both orchestrator.py's
    mid-run PROBE_QUESTION path and eda_agent.py's agentic-EDA exploration loop.
    Returns (result, run_attempts)."""
    result_path = probe_dir / 'probe_result.json'
    run_attempts = []
    display.run_start('diagnostic probe', probe_timeout)
    t0 = time.time()
    result = run_probe_candidate(probe_dir, data_dir, result_path, probe_timeout)
    display.run_end(t0, result['ok'])
    run_attempts.append({'attempt': 0, 'ok': result['ok'], 'error': result.get('error')})

    repairs_left = max_repairs
    while not result['ok'] and repairs_left > 0:
        repairs_left -= 1
        current_probe = {'probe.py': (probe_dir / 'probe.py').read_text(encoding='utf-8')}
        repair_prompt = pt.build_repair_prompt(current_probe, question, result['error'])
        attempt, tries = call_llm_with_retry(system_prompt, repair_prompt, model, max_budget_usd,
                                              label=f'repairing probe after run failure ({repairs_left + 1} left)',
                                              timeout=propose_timeout)
        llm_calls.extend({'cost_usd': t.get('cost_usd'), 'ok': t['ok'], 'error': t.get('error')} for t in tries)
        if not attempt['ok']:
            run_attempts.append({'attempt': len(run_attempts), 'ok': False,
                                  'error': f'repair LLM call failed: {attempt["error"]}'})
            break
        repaired_files = pt.parse_response(attempt['text'])['files']
        for fname, content in repaired_files.items():
            if fname in allowed_files:
                (probe_dir / fname).write_text(content, encoding='utf-8')
        display.run_start('diagnostic probe (repaired)', probe_timeout)
        t0 = time.time()
        result = run_probe_candidate(probe_dir, data_dir, result_path, probe_timeout)
        display.run_end(t0, result['ok'])
        run_attempts.append({'attempt': len(run_attempts), 'ok': result['ok'], 'error': result.get('error')})
    return result, run_attempts
