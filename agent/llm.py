"""Thin wrapper around headless `claude -p` — the agent's only LLM access point.

Uses the same login/session as interactive Claude Code (no separate API key) and caps
spend per call (--max-budget-usd). Three entry points, deliberately kept separate rather
than one function with a `tools` parameter, so the security boundary between them can
never be blurred by a careless call-site change:

- `call_claude()` — disables all tools (--tools ""), so the sub-call is a pure text
  completion. This is the ONLY function used by the propose/repair/probe path
  (prompt_templates.SYSTEM_PROMPT) — that path's raw text output is applied directly
  as executed code, so it must never be able to take an action of its own.
- `call_claude_research()` — grants ONLY WebSearch + WebFetch (never Bash/PowerShell/
  code-execution), for agent/web_research.py's research-only role. Its output is never
  applied as code either; see that module's docstring for the full rationale.
- `call_claude_explore()` — grants ONLY Read/Grep/Glob (never Bash/PowerShell/Edit/Write/
  code-execution), confined to this repo's own root, for agent/repo_explore.py's
  read-only "what already exists in this project" role. Same non-actionable-output
  guarantee as the other two; see that module's docstring for the full rationale.
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
DEFAULT_RESEARCH_TIMEOUT_S = 120
DEFAULT_EXPLORE_TIMEOUT_S = 90

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _call_claude(tool_args, system_prompt, user_prompt, model, max_budget_usd, timeout, cwd=None):
    """Shared subprocess plumbing for all entry points below. `tool_args` is the list
    of extra CLI args that controls what the sub-call is allowed to do — the main thing
    that differs between call_claude()/call_claude_research()/call_claude_explore().
    `cwd`, when given, is where the subprocess runs — this is what `--restricted`'s
    "confines file-tool access to the working directory" guarantee actually anchors to,
    so a tool-granted role (research, explore) should always pass one explicitly rather
    than inheriting whatever directory happened to invoke the orchestrator.
    Returns a dict:
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
            *tool_args,
            '--model', model,
            '--max-budget-usd', str(max_budget_usd),
            '--system-prompt-file', sys_prompt_path,
            '--no-session-persistence',
        ]
        try:
            proc = subprocess.run(cmd, input=user_prompt, capture_output=True, text=True,
                                   timeout=timeout, encoding='utf-8', errors='replace', cwd=cwd)
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


def call_claude(system_prompt, user_prompt, model=DEFAULT_MODEL,
                 max_budget_usd=DEFAULT_MAX_BUDGET_USD, timeout=DEFAULT_TIMEOUT_S):
    """Pure text completion — no tools, no filesystem/network access of its own. This is
    the only function the propose/repair/probe path may call (see module docstring)."""
    return _call_claude(['--tools', ''], system_prompt, user_prompt, model, max_budget_usd, timeout)


def call_claude_research(system_prompt, user_prompt, model=DEFAULT_MODEL,
                          max_budget_usd=DEFAULT_MAX_BUDGET_USD, timeout=DEFAULT_RESEARCH_TIMEOUT_S):
    """WebSearch + WebFetch only — no Bash/PowerShell/REPL or any other code-execution
    tool, ever. `--restricted` strips those unconditionally and also ignores project/
    user settings + hooks (so a malicious fetched page can't smuggle in a tool-config
    change) and confines file-tool access to the working directory; `--tools` then
    names exactly the two tools this role needs, and `--allowedTools` pre-approves them
    so a non-interactive -p call doesn't hang on a permission prompt. Used only by
    agent/web_research.py — never by the propose/repair/probe path, which must stay on
    call_claude() above so its output can never come from anything but a pure text
    completion."""
    return _call_claude(['--restricted', '--tools', 'WebSearch,WebFetch', '--allowedTools', 'WebSearch,WebFetch'],
                         system_prompt, user_prompt, model, max_budget_usd, timeout)


def call_claude_explore(system_prompt, user_prompt, model=DEFAULT_MODEL,
                         max_budget_usd=DEFAULT_MAX_BUDGET_USD, timeout=DEFAULT_EXPLORE_TIMEOUT_S):
    """Read/Grep/Glob only — no Bash/PowerShell/Edit/Write, ever, so this role can look at
    the project's own repo but never act on it. Same `--restricted` guarantee as
    call_claude_research() above (strips code-execution unconditionally, ignores project/
    user settings + hooks, confines file-tool access to the working directory) — and here
    that working directory is pinned explicitly to REPO_ROOT via `cwd`, rather than relying
    on wherever the orchestrator happened to be launched from, since REPO_ROOT is exactly
    the boundary this role must stay inside. Used only by agent/repo_explore.py — never by
    the propose/repair/probe path, which must stay on call_claude() above."""
    return _call_claude(['--restricted', '--tools', 'Read,Grep,Glob', '--allowedTools', 'Read,Grep,Glob'],
                         system_prompt, user_prompt, model, max_budget_usd, timeout, cwd=str(REPO_ROOT))
