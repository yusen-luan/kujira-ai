"""Thin wrapper around headless `claude -p` — the agent's only LLM access point.

Uses the same login/session as interactive Claude Code (no separate API key),
disables all tools (--tools "") so the sub-call is a pure text completion the
orchestrator can't lose control of, and caps spend per call (--max-budget-usd).
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_MODEL = 'sonnet'
DEFAULT_MAX_BUDGET_USD = 0.50
DEFAULT_TIMEOUT_S = 180


def _resolve_claude_bin():
    """Find a real, directly-executable claude binary.

    CLAUDE_CODE_EXECPATH (if set) only exists inside an active Claude Code
    session — it is NOT a persistent environment variable, so a fresh IDE/
    terminal process won't have it. In that case `claude` on PATH resolves to
    a .cmd/.ps1 npm shim on Windows, which subprocess (shell=False) can't
    exec directly. Fall back to the real .exe sitting next to it on disk,
    at the standard npm global-install layout.
    Returns None if nothing runnable could be found.
    """
    exe = os.environ.get('CLAUDE_CODE_EXECPATH')
    if exe and os.path.isfile(exe):
        return exe

    found = shutil.which('claude')
    if not found:
        return None
    if not found.lower().endswith(('.cmd', '.bat', '.ps1')):
        return found  # already a real executable (e.g. on Linux/macOS/WSL)

    candidate = Path(found).parent / 'node_modules' / '@anthropic-ai' / 'claude-code' / 'bin' / 'claude.exe'
    return str(candidate) if candidate.is_file() else None


CLAUDE_BIN = _resolve_claude_bin()


def call_claude(system_prompt, user_prompt, model=DEFAULT_MODEL,
                 max_budget_usd=DEFAULT_MAX_BUDGET_USD, timeout=DEFAULT_TIMEOUT_S):
    """Returns a dict:
      ok=True  -> {'ok': True, 'text': str, 'cost_usd': float, 'duration_ms': int}
      ok=False -> {'ok': False, 'error': str}
    Never raises — callers use 'ok' to decide whether to retry/abort.
    """
    if not CLAUDE_BIN:
        return {'ok': False, 'error': 'could not locate a runnable claude executable — '
                                       'is Claude Code installed and on PATH?'}

    # System prompt via --system-prompt-file and user prompt via stdin (rather than
    # `-p <text>` / `--system-prompt <text>`) because Windows' CreateProcess has a
    # ~32K-character total command-line limit — large prompts (e.g. the model axis's,
    # which inlines several hundred lines of reference code) blow past that and fail
    # with WinError 206 "filename or extension is too long". Neither the system-prompt
    # file nor stdin has that ceiling.
    fd, sys_prompt_path = tempfile.mkstemp(suffix='.txt', prefix='claude_sysprompt_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(system_prompt)
        cmd = [
            CLAUDE_BIN, '-p',
            '--output-format', 'json',
            '--tools', '',
            '--model', model,
            '--max-budget-usd', str(max_budget_usd),
            '--system-prompt-file', sys_prompt_path,
            '--no-session-persistence',
        ]
        try:
            proc = subprocess.run(cmd, input=user_prompt, capture_output=True, text=True,
                                   timeout=timeout, encoding='utf-8', errors='replace')
        except subprocess.TimeoutExpired:
            return {'ok': False, 'error': f'llm call timed out after {timeout}s'}
    finally:
        try:
            os.remove(sys_prompt_path)
        except OSError:
            pass

    if proc.returncode != 0:
        return {'ok': False, 'error': f'claude CLI exited {proc.returncode}: {proc.stderr[-2000:]}'}

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {'ok': False, 'error': f'claude CLI produced non-JSON output: {proc.stdout[-2000:]}'}

    if data.get('is_error'):
        return {'ok': False, 'error': f"claude reported an error: {data.get('result')}"}

    return {
        'ok': True,
        'text': data.get('result', ''),
        'cost_usd': data.get('total_cost_usd', 0.0),
        'duration_ms': data.get('duration_ms', 0),
    }
