"""Terminal presentation helpers for orchestrator.py — the only place that decides how
things look. Plain functions, no state beyond a quiet-mode toggle and color detection,
so orchestrator.py's control flow stays readable instead of interleaved with
formatting logic.

Goal (per user request): make a plain `python agent/orchestrator.py` run legible in a
terminal at all times — which phase is running, every LLM call (what it was asked to
do and a preview of what it said back), live per-epoch training progress instead of a
silent multi-minute subprocess, and a running hypothesis tree showing how the two axes
have branched and why. Not a TUI/curses redraw — plain scrolling output, same category
of tool as a build log, just narrated.
"""
import sys
import time

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


def diagnosis_line(category, rationale):
    print(f'  {_magenta("diagnosis")} → {_bold(category)}  {_trunc(rationale, 100)}')


def probe_line(status, question):
    tag = _green('answered') if status == 'answered' else _red('FAILED')
    print(f'  {_magenta("probe")} {tag}  "{_trunc(question, 90)}"')


def converged(eps, n):
    print(f'  {_yellow("●")} converged: no >{eps} improvement over the last {n} nodes — stopping early')


def _trunc(s, n):
    if not s:
        return ''
    s = str(s).replace('\n', ' ').strip()
    return s if len(s) <= n else s[:n - 1] + '…'


# ---------------- hypothesis tree ----------------

def _node_line(h):
    status = h.get('status')
    tag = f'[{h["iter"]}]'
    if status == 'accepted':
        return f'{tag} {_green("accepted")}  {h["primary"]:.4f}  "{_trunc(h.get("hypothesis"), 70)}"'
    if status == 'rejected':
        return f'{tag} {_yellow("rejected")}  {h["primary"]:.4f}  "{_trunc(h.get("hypothesis"), 70)}"'
    if status == 'diagnosed':
        return f'{tag} {_magenta("diagnosis")}  category={h.get("category")}'
    if status == 'answered':
        return f'{tag} {_magenta("probe")}  "{_trunc(h.get("question"), 70)}"'
    if status == 'failed' and h.get('question') is not None:
        return f'{tag} {_red("probe FAILED")}  "{_trunc(h.get("question"), 70)}"'
    return f'{tag} {_red("FAILED")}  "{_trunc(h.get("hypothesis"), 70)}"'


def render_tree(baseline_primary, axes_order, history):
    """Groups history entries by root axis (feature/model/synthesis -- a
    '<axis>_diagnosis'/'<axis>_probe' entry's root is `axis`) and renders them as a
    branch each, in chronological order within a branch. Called after every node so
    re-printing it gives a live-updating picture of how the run has branched."""
    groups, order = {}, []
    for h in history:
        axis = h.get('axis') or 'synthesis'
        root = axis.split('_')[0] if not axis.startswith('synthesis') else 'synthesis'
        groups.setdefault(root, []).append(h)
        if root not in order:
            order.append(root)
    root_order = [a for a in axes_order if a in groups] + [r for r in order if r not in axes_order]

    lines = [_bold('hypothesis tree so far:'), f'  {_dim("●")} baseline   primary {baseline_primary:.4f}']
    for i, root in enumerate(root_order):
        is_last_root = i == len(root_order) - 1
        root_branch = '  └─' if is_last_root else '  ├─'
        lines.append(f'{root_branch} {_bold(root)}')
        entries = groups[root]
        child_prefix = '     ' if is_last_root else '  │  '
        for j, h in enumerate(entries):
            is_last = j == len(entries) - 1
            connector = '└─' if is_last else '├─'
            lines.append(f'{child_prefix}{connector} {_node_line(h)}')
    print('\n'.join(lines))
