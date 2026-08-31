"""Bounded multi-turn coding session -- v4 roadmap Phase 4, the last planned piece. Every
other propose-turn mode (hypothesis/probe/research/explore/eda_round) either has zero tools
(the propose/repair/probe path itself) or grants only read-only lookup tools (research/
explore). This is the first role with real Edit/Write access: instead of guessing a whole
replacement file body in one shot and finding out if it runs via the repair loop afterward,
the model can Read the current data.py/baseline.py, edit them directly, and iterate before
handing off -- worth the extra cost mainly for a change gnarly enough that reading-while-
editing genuinely helps.

How this stays safe, concretely:
- Runs through llm.call_claude_code(), which grants ONLY Read/Grep/Glob/Edit/Write (never
  Bash/PowerShell/REPL or any other code-execution tool), confined via `cwd` to exactly one
  candidate node directory. See llm.py's module docstring for why Bash is deliberately never
  granted here: verified live during Phase 4 planning that --restricted's cwd-confinement
  guarantee is real for Read/Edit/Write but does NOT extend to Bash, which can reach anywhere
  the OS user can regardless of cwd/--add-dir/--restricted.
- `cwd` is always the orchestrator's own per-node scratch directory (agent/runs/node_N/,
  already snapshotted from the current best/ before this module is ever called) -- never
  REPO_ROOT and never a data directory. With no Bash grant, this cwd confinement of the file
  tools IS the entire safety boundary this role runs inside.
- Its output is NEVER text parsed back into a file (unlike the propose/repair/probe path's
  fenced-code-block contract) -- the file changes are already on disk via its own tool calls
  by the time the call returns. The only text this module hands back to orchestrator.py is
  the session's final REFLECTION/HYPOTHESIS/EXPECTED_EFFECT/LEVER_CATEGORY message, parsed
  with the exact same prompt_templates.parse_response() the plain hypothesis path uses (its
  fallback to mode='hypothesis' on a files-free, question-free response is exactly what a
  code session's final text looks like -- see that function's docstring).
- One question per call, one call per orchestrator node, hard-capped for the whole run by
  --code_session_budget (a separate pool from every other budget) -- see
  orchestrator.py's run_iteration(). No call_llm_with_retry wrapper here (unlike the propose/
  repair/probe path) -- matches web_research.py/repo_explore.py's precedent of a single call
  for a narrow-tool role; a failed call is just a failed node, same as those.
"""
import prompt_templates as pt
import llm


def run_code_session(iter_dir, best_code, history, question, model=llm.DEFAULT_MODEL,
                      max_budget_usd=llm.DEFAULT_MAX_BUDGET_USD,
                      timeout=llm.DEFAULT_CODE_SESSION_TIMEOUT_S, torch_available=True):
    """Runs one bounded multi-turn session with cwd=iter_dir (already populated with the
    current best data.py/baseline.py by the caller, via orchestrator.snapshot_node_dir).
    `question` is the propose call's own CODE_SESSION_REQUEST text -- passed through as this
    session's starting context/brief. Never raises. Returns {'ok': bool, 'text': str,
    'cost_usd': float, 'error': str|None} -- on ok=True, `text` is the session's final
    REFLECTION/HYPOTHESIS/EXPECTED_EFFECT/LEVER_CATEGORY message for orchestrator.py to parse
    with prompt_templates.parse_response(); the actual file changes are already sitting in
    iter_dir regardless of what parses from that text."""
    system_prompt = pt.build_code_session_system_prompt(torch_available)
    user_prompt = pt.build_code_session_prompt(best_code, history, question)
    result = llm.call_claude_code(system_prompt, user_prompt, cwd=str(iter_dir), model=model,
                                   max_budget_usd=max_budget_usd, timeout=timeout)
    if not result['ok']:
        return {'ok': False, 'text': '', 'cost_usd': 0.0, 'error': result['error']}
    return {'ok': True, 'text': result['text'], 'cost_usd': result.get('cost_usd', 0.0), 'error': None}
