"""Read-only repo exploration — a third, narrowly-scoped LLM role, separate from both
the propose/repair/probe path and agent/web_research.py. Added because the propose role
(agent/llm.py's call_claude(), zero tools) can only ever see what's pasted into its
prompt — it has no way to notice that, say, workspace/ablation_features.py already
answers a question it's about to spend an iteration re-deriving. This module lets it ask.

How this stays safe and cheap, concretely (mirrors agent/web_research.py's approach,
substituting "this repo" for "the web"):
- Runs through `llm.call_claude_explore()`, which grants ONLY Read/Grep/Glob (no
  Bash/PowerShell/Edit/Write/code-execution, ever) confined to REPO_ROOT via `cwd` — see
  llm.py's module docstring. The propose/repair/probe call is completely untouched and
  still has zero tool access; this module's output is never applied as code, only ever
  saved as a markdown note.
- The model is asked for exactly ONE fixed-format note per question (same shape as the
  hand-curated agent/literature/*.md notes and agent/web_research.py's saved notes), and
  MUST cite the actual file path(s) it read.
- Every cited path is validated here, in code, against the real filesystem (must exist,
  must resolve under REPO_ROOT) — not left to the model's own say-so, same principle as
  web_research.py's domain-allowlist check. A note citing a nonexistent or out-of-repo
  path is discarded outright, regardless of how plausible its content reads.
- One question per call, one call per orchestrator node, hard-capped for the whole run
  by --explore_budget (a separate pool from --web_research_budget/--iterations) — see
  orchestrator.py's run_iteration().
- Accepted notes are saved to agent/runs/repo_notes/ (gitignored, like agent/runs/
  web_literature/) and merged into retrieval by rag.py, tagged `source: repo_explore` so
  the propose prompt can tell "found by reading this project's own repo" apart from both
  curated literature and live web-research notes.
"""
import re
from pathlib import Path

import llm

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent
RUNS_DIR = AGENT_DIR / 'runs'
REPO_NOTES_DIR = RUNS_DIR / 'repo_notes'

SYSTEM_PROMPT = """You are a research assistant helping an ML engineering agent understand what \
already exists in its OWN project repository — other scripts, prior analyses, unused reference \
code, design docs — before it spends an iteration re-deriving something that's already sitting on \
disk. You have read-only Read/Grep/Glob tools, confined to this repository's root. You do NOT \
have Bash/PowerShell/Edit/Write or any other way to execute or modify anything, and nothing you \
produce here is ever run — only read by another LLM call later.

This is about what EXISTS IN THIS REPO right now (code, scripts, docs, past analysis) — not a \
question about the raw dataset's contents (a different mechanism, a numeric probe, handles that) \
and not a question about published external methods (a different mechanism, live web search, \
handles that). If the question you're asked doesn't fit "what's already in this codebase," say so \
in your note's body rather than guessing at something else.

Every non-obvious claim in your note must trace back to a file you actually read this turn — cite \
its path (relative to the repo root, e.g. "workspace/ablation_features.py") in the note body. Do \
not describe a file's contents from its name alone; open it.

Answer with EXACTLY ONE note in this format, nothing else — no preamble, no commentary before or \
after it:

---
title: <short descriptive title>
citation: <the repo-relative file path(s) this note is based on, comma-separated if more than one>
tags: <comma-separated keywords useful for later retrieval>
---
<4-8 sentences: what you found, in which file(s), and why it's relevant to the current ML \
pipeline (data.py/baseline.py). Name every file path you drew on inline in the text, not just in \
the citation line above.>

If you genuinely found nothing relevant after looking, respond with EXACTLY this and nothing else:
NOTHING_RELEVANT_FOUND: <one sentence on what you looked at>"""


def build_explore_prompt(question):
    return f"""Question: {question}

Look through this repository now (Read/Grep/Glob only) and answer using the fixed note format \
from your instructions."""


_NOTE_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL | re.MULTILINE)
_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+")


def _cited_paths_exist(citation, body):
    """Every path named in `citation` (comma-separated) must exist on disk under
    REPO_ROOT. Also scans the body for path-shaped tokens ending in a file extension
    that look like they're claiming to be repo files (contain a '/') and, if any such
    token doesn't resolve, rejects the note too -- this catches a note that names a real
    file in `citation` but then fabricates a second, different path in the prose."""
    candidates = [c.strip() for c in citation.split(',') if c.strip()]
    candidates += [tok for tok in _PATH_TOKEN_RE.findall(body) if '/' in tok or '\\' in tok]
    if not candidates:
        return False
    for raw in candidates:
        p = (REPO_ROOT / raw.replace('\\', '/')).resolve()
        try:
            p.relative_to(REPO_ROOT.resolve())
        except ValueError:
            return False  # escapes the repo root -- reject regardless of existence
        if not p.exists():
            return False
    return True


def parse_explore_response(text):
    """Returns None if nothing relevant was found, the response didn't parse, or any
    cited/mentioned path doesn't actually exist under the repo root (checked here in
    code -- never trusted purely on the model's say-so). Otherwise returns
    {'title', 'citation', 'tags', 'body'}."""
    text = text.strip()
    if not text or text.startswith('NOTHING_RELEVANT_FOUND'):
        return None
    m = _NOTE_RE.search(text)  # search, not match -- a stray preamble sentence before the
                                # frontmatter (seen in practice) would otherwise make a
                                # perfectly valid note silently unparseable
    if not m:
        return None
    header, body = m.group(1), m.group(2).strip()
    meta = {}
    for line in header.strip().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    citation = meta.get('citation', '')
    if not body or not _cited_paths_exist(citation, body):
        return None
    return {'title': meta.get('title', 'untitled') or 'untitled', 'citation': citation,
            'tags': meta.get('tags', ''), 'body': body}


def _slug(title):
    s = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
    return s[:60] or 'note'


def save_note(note, node_id):
    """`source: repo_explore` must be exactly that string -- rag.py's
    build_grounding_context() does an exact-match check to pick the right provenance
    label, same convention as web_research.py's `source: web`."""
    REPO_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    path = REPO_NOTES_DIR / f"{node_id}_{_slug(note['title'])}.md"
    text = (f"---\ntitle: {note['title']}\ncitation: {note['citation']}\n"
            f"tags: {note['tags']}\nsource: repo_explore\nfound_at_node: {node_id}\n---\n"
            f"{note['body']}\n")
    path.write_text(text, encoding='utf-8')
    return path


def run_explore(question, node_id, model=llm.DEFAULT_MODEL, max_budget_usd=llm.DEFAULT_MAX_BUDGET_USD,
                 timeout=llm.DEFAULT_EXPLORE_TIMEOUT_S):
    """One explore call. Never raises -- a rejected/failed result just means this node
    spent its explore-budget slot and found nothing usable, same as a failed probe or
    research call. Returns {'ok': bool, 'question': str, 'note': dict|None,
    'rejected_reason': str|None, 'cost_usd': float, 'saved_path': str|None}."""
    prompt = build_explore_prompt(question)
    result = llm.call_claude_explore(SYSTEM_PROMPT, prompt, model=model,
                                      max_budget_usd=max_budget_usd, timeout=timeout)
    if not result['ok']:
        return {'ok': False, 'question': question, 'note': None,
                'rejected_reason': f"LLM call failed: {result['error']}",
                'cost_usd': 0.0, 'saved_path': None}

    note = parse_explore_response(result['text'])
    if note is None:
        return {'ok': False, 'question': question, 'note': None,
                'rejected_reason': 'no note with a verifiable repo-file citation was returned',
                'cost_usd': result.get('cost_usd', 0.0), 'saved_path': None}

    path = save_note(note, node_id)
    return {'ok': True, 'question': question, 'note': note, 'rejected_reason': None,
            'cost_usd': result.get('cost_usd', 0.0), 'saved_path': str(path)}
