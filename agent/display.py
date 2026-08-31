"""Terminal presentation helpers for orchestrator.py — the only place that decides how
things look. Plain functions, no state beyond a quiet-mode toggle and color detection,
so orchestrator.py's control flow stays readable instead of interleaved with
formatting logic.

Goal (per user request): make a plain `python agent/orchestrator.py` run legible in a
terminal at all times — which phase is running, every LLM call (what it was asked to
do and a preview of what it said back), live per-epoch training progress instead of a
silent multi-minute subprocess, and a running history of the iteration chain showing
what was tried and why. Not a TUI/curses redraw — plain scrolling output, same category
of tool as a build log, just narrated.
"""
import sys
import time

# Windows consoles that aren't UTF-8 configured (legacy cmd.exe/PowerShell, some CI
# runners) default Python's stdout to the system codepage (e.g. cp1252), which can't
# encode the arrow/bullet symbols below and crashes with UnicodeEncodeError on the
# first one printed. Force UTF-8 with a safe fallback so this module works regardless
# of the terminal's configured codepage -- worst case on a truly incompatible terminal
# is a '?' in place of a symbol, never a crash. No-op (and harmless) on terminals that
# are already UTF-8, which is the common case (WSL/Linux/macOS terminals, Windows
# Terminal).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

_USE_COLOR = sys.stdout.isatty()
_QUIET = False


def set_quiet(quiet):
    global _QUIET
    _QUIET = quiet


def _c(code, s):
    return f"\x1b[{code}m{s}\x1b[0m" if _USE_COLOR else s


def _bold(s):
    return _c('1', s)


def _dim(s):
    return _c('2', s)


def _green(s):
    return _c('32', s)


def _red(s):
    return _c('31', s)


def _yellow(s):
    return _c('33', s)


def _cyan(s):
    return _c('36', s)


def _magenta(s):
    return _c('35', s)


def banner(msg):
    print()
    print(_bold(f'=== {msg} ==='))


def step(msg):
    print(f'  {msg}')


def phase(msg):
    """A one-off status line for a non-LLM, non-training step (EDA, literature
    retrieval, baseline reproduction, convergence checks, etc.)."""
    print(f'  {_cyan("·")} {msg}')


def llm_call_start(label):
    print(f'  {_cyan("→")} calling claude — {label} ...')
    return time.time()


def llm_call_end(t0, attempt):
    dt = time.time() - t0
    if attempt['ok']:
        print(f'  {_cyan("←")} responded in {dt:.1f}s (${attempt.get("cost_usd", 0.0):.3f})')
    else:
        print(f'  {_red("←")} FAILED after {dt:.1f}s: {attempt["error"]}')


def retrying():
    print(f'  {_yellow("↻")} retrying...')


def field(name, value, width=15):
    """One labeled preview line for a parsed LLM response field (reflection,
    hypothesis, expected_effect, category, rationale, ...). Skips empty values
    silently (e.g. a first-ever iteration on an axis has no reflection yet)."""
    if not value:
        return
    text = str(value).replace('\n', ' ').strip()
    if len(text) > 320:
        text = text[:319] + '…'
    print(f'      {_dim(f"{name:>{width}}:")} {text}')


def run_start(label, timeout):
    print(f'  {_cyan("→")} running {label} (timeout {timeout}s)...')


def run_line(line):
    if _QUIET:
        return
    print(f'      {_dim("│")} {line}')


def run_end(t0, ok, timed_out=False):
    dt = time.time() - t0
    if timed_out:
        print(f'  {_red("←")} run TIMED OUT after {dt:.0f}s')
    elif ok:
        print(f'  {_cyan("←")} run finished in {dt:.0f}s')
    else:
        print(f'  {_red("←")} run failed after {dt:.0f}s')


def result_line(status, hypothesis, primary, prev_best, error_summary=None):
    hyp = _trunc(hypothesis, 90)
    if status == 'accepted':
        print(f'  {_green("ACCEPTED")}  primary {primary:.4f} (was {prev_best:.4f})  "{hyp}"')
    elif status == 'rejected':
        print(f'  {_yellow("rejected")}  primary {primary:.4f} (best stays {prev_best:.4f})  "{hyp}"')
    else:
        print(f'  {_red("FAILED")}    "{hyp}"  — {error_summary}')


def probe_line(status, question):
    tag = _green('answered') if status == 'answered' else _red('FAILED')
    print(f'  {_magenta("probe")} {tag}  "{_trunc(question, 90)}"')


def research_line(status, question, note_title=None):
    if status == 'researched':
        print(f'  {_magenta("web research")} {_green("found")}  "{_trunc(note_title, 60)}"  '
              f'for "{_trunc(question, 60)}"')
    elif status == 'research_denied':
        print(f'  {_magenta("web research")} {_yellow("denied (budget exhausted)")}  "{_trunc(question, 90)}"')
    else:
        print(f'  {_magenta("web research")} {_red("no reliable source")}  "{_trunc(question, 90)}"')


def explore_line(status, question, note_title=None):
    if status == 'explored':
        print(f'  {_magenta("repo explore")} {_green("found")}  "{_trunc(note_title, 60)}"  '
              f'for "{_trunc(question, 60)}"')
    elif status == 'explore_denied':
        print(f'  {_magenta("repo explore")} {_yellow("denied (budget exhausted)")}  "{_trunc(question, 90)}"')
    else:
        print(f'  {_magenta("repo explore")} {_red("nothing relevant found")}  "{_trunc(question, 90)}"')


def eda_round_line(status, question, n_probes=None):
    if status == 'eda_round':
        print(f'  {_magenta("eda round")} {_green(f"{n_probes} new finding(s)")}  '
              f'for "{_trunc(question, 60)}"')
    elif status == 'eda_round_denied':
        print(f'  {_magenta("eda round")} {_yellow("denied (budget exhausted)")}  "{_trunc(question, 90)}"')
    else:
        print(f'  {_magenta("eda round")} {_red("nothing new found")}  "{_trunc(question, 90)}"')


def code_session_line(status, question):
    """Only ever called for status == 'code_session_denied' -- an ATTEMPTED code session's
    outcome (accepted/rejected/failed) already goes through result_line() like any other
    hypothesis, so this only needs the one 'request denied' case (mirrors the *_denied
    branch of research_line/explore_line/eda_round_line above)."""
    print(f'  {_magenta("code session")} {_yellow("denied (budget exhausted)")}  "{_trunc(question, 90)}"')


def converged(eps, n):
    print(f'  {_yellow("●")} converged: no >{eps} improvement over the last {n} nodes — stopping early')


def _trunc(s, n):
    if not s:
        return ''
    s = str(s).replace('\n', ' ').strip()
    return s if len(s) <= n else s[:n - 1] + '…'


# ---------------- run history (v2: single linear chain, no more axis branches) ----------------

def _sweep_tag(h):
    sweep = h.get('sweep')
    if not sweep:
        return ''
    n_ok = sum(1 for t in sweep['trials'] if t['ok'])
    return f'  [swept {sweep["param"]}={sweep["best_value"]} best of {n_ok}/{len(sweep["trials"])}]'


def _variants_tag(h):
    variants = h.get('variants')
    if not variants:
        return ''
    n_ok = sum(1 for v in variants if v['ok'])
    return f'  [raced {len(variants)} variants, winner v{h.get("winning_variant")}, {n_ok}/{len(variants)} ran ok]'


def _code_session_tag(h):
    return '  [via code session]' if h.get('authored_via') == 'code_session' else ''


def _node_line(h):
    status = h.get('status')
    tag = f'[{h["iter"]}]'
    if status == 'accepted':
        return (f'{tag} {_green("accepted")}  {h["primary"]:.4f}  "{_trunc(h.get("hypothesis"), 70)}"'
                 f'{_sweep_tag(h)}{_variants_tag(h)}{_code_session_tag(h)}')
    if status == 'rejected':
        return (f'{tag} {_yellow("rejected")}  {h["primary"]:.4f}  "{_trunc(h.get("hypothesis"), 70)}"'
                 f'{_sweep_tag(h)}{_variants_tag(h)}{_code_session_tag(h)}')
    if status == 'answered':
        return f'{tag} {_magenta("probe")}  "{_trunc(h.get("question"), 70)}"'
    if status == 'failed' and h.get('question') is not None:
        return f'{tag} {_red("probe FAILED")}  "{_trunc(h.get("question"), 70)}"'
    if status == 'researched':
        return (f'{tag} {_magenta("web research")}  found "{_trunc(h.get("note_title"), 40)}" for '
                 f'"{_trunc(h.get("research_question"), 40)}"')
    if status in ('research_failed', 'research_denied'):
        why = 'denied' if status == 'research_denied' else 'no source'
        return f'{tag} {_red(f"web research {why}")}  "{_trunc(h.get("research_question"), 70)}"'
    # explored/explore_*, eda_round/eda_round_*, code_session_denied: pre-existing gap found
    # while adding the branches above -- these previously had no case here at all and fell
    # straight through to the generic FAILED line below even on a genuine success, unrelated
    # to v4 roadmap Phase 4 but directly adjacent, so fixed alongside rather than left in
    # place (research_line's shape above already got this right; these three families didn't).
    if status == 'explored':
        return (f'{tag} {_magenta("repo explore")}  found "{_trunc(h.get("note_title"), 40)}" for '
                 f'"{_trunc(h.get("explore_question"), 40)}"')
    if status in ('explore_failed', 'explore_denied'):
        why = 'denied' if status == 'explore_denied' else 'no source'
        return f'{tag} {_red(f"repo explore {why}")}  "{_trunc(h.get("explore_question"), 70)}"'
    if status == 'eda_round':
        return f'{tag} {_magenta("eda round")}  "{_trunc(h.get("eda_round_question"), 70)}"'
    if status in ('eda_round_failed', 'eda_round_denied'):
        why = 'denied' if status == 'eda_round_denied' else 'nothing new'
        return f'{tag} {_red(f"eda round {why}")}  "{_trunc(h.get("eda_round_question"), 70)}"'
    if status == 'code_session_denied':
        return f'{tag} {_red("code session denied")}  "{_trunc(h.get("code_session_question"), 70)}"'
    return f'{tag} {_red("FAILED")}  "{_trunc(h.get("hypothesis"), 70)}"'


def render_history(baseline_primary, history, label='baseline'):
    """Flat, chronological list -- there's only one chain now (v2 dropped the v1
    feature/model axis-tree), so no grouping/branching is needed. Called after every
    node so re-printing it gives a live-updating picture of the run. `label` is
    'baseline' only when node 0 actually reproduced workspace/'s pristine files --
    with --resume (the default) after any prior accept, node 0 re-runs the carried-over
    best/ state instead, which node0_label() in orchestrator.py detects and labels
    'last best' so this line doesn't misleadingly claim to be the organizer's reference."""
    lines = [_bold('run history so far:'), f'  {_dim("●")} {label:10s} primary {baseline_primary:.4f}']
    for h in history:
        lines.append(f'  {_node_line(h)}')
    print('\n'.join(lines))
