"""Live web research — a separate, narrowly-scoped role from the propose/repair/probe
LLM path, added so the agent can draw on published methods it doesn't already have in
the static curated corpus (agent/literature/), without reopening the risks that kept
this out of v1/v2 (see agent_notes/orchestrator.md's "Literature RAG step" section for
that history): non-determinism, prompt-injection into executed code, low-quality open-
web results, and unbounded cost.

How this stays safe and cheap, concretely:
- Runs through `llm.call_claude_research()`, which grants ONLY WebSearch/WebFetch (no
  Bash/PowerShell/code-execution, ever) — see llm.py's module docstring. The propose/
  repair/probe call (prompt_templates.SYSTEM_PROMPT, via llm.call_claude()) is
  completely untouched and still has zero tool access; this module's output is never
  applied as code, only ever saved as a markdown note.
- The model is asked for exactly ONE fixed-format note per question (same shape as the
  hand-curated agent/literature/*.md notes), and MUST name a single `source_url`.
- That citation is validated here, in code, against ALLOWED_DOMAINS — not left to the
  model's own judgment — before anything is written to disk. An untrusted or missing
  citation means the note is discarded, regardless of how plausible its content reads.
- One question per call, one call per orchestrator node, hard-capped for the whole run
  by --web_research_budget (a separate pool from --iterations, like the old
  --diagnosis_budget) — see orchestrator.py's run_iteration().
- Accepted notes are saved to agent/runs/web_literature/ (NOT agent/literature/, which
  stays the human-curated, checked-in corpus) and merged into retrieval by rag.py,
  tagged `source: web` so the propose prompt can tell curated methods apart from ones
  the agent found live this run.
"""
import re
from pathlib import Path

import llm

AGENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = AGENT_DIR / 'runs'
WEB_LITERATURE_DIR = RUNS_DIR / 'web_literature'

# Deliberately conservative and recsys/ML-specific -- a citation from anywhere else is
# rejected outright in parse_research_response(), no matter how relevant the text reads.
# Matches the kind of source agent/literature/'s own notes were built from (see that
# corpus's citations): arXiv, official repos, ACM/IEEE, paperswithcode, and a short list
# of named industry research blogs -- not forums, SEO content, or unattributed pages.
ALLOWED_DOMAINS = (
    'arxiv.org', 'paperswithcode.com', 'dl.acm.org', 'ieeexplore.ieee.org',
    'openreview.net', 'github.com', 'kuairand.com', 'dblp.org',
    'ai.googleblog.com', 'research.google', 'developers.google.com',
    'ai.meta.com', 'engineering.fb.com', 'research.facebook.com',
)

SYSTEM_PROMPT = """You are a research assistant helping an ML engineering agent that is stuck \
find a specific published method or fact. You have live web search and page-fetch tools. You do \
NOT write or execute any code, and nothing you produce here is ever run — only read by another LLM \
call later.

Trust only these kinds of sources, in this order of preference: arXiv abstract/PDF pages, official \
project GitHub repositories, ACM/IEEE Digital Library pages, paperswithcode.com, OpenReview, and \
named industry research blogs (Google AI Blog, Meta/FAIR AI Blog). Do NOT cite forum posts, SEO \
content farms, unattributed aggregator pages, or anything you cannot trace to a named, identifiable \
publisher. If you cannot find a trustworthy source for the question asked, say so explicitly rather \
than citing a weak one — a missed question costs nothing; a bad citation gets used as if it were \
verified fact.

Answer with EXACTLY ONE note in this format, nothing else — no preamble, no commentary before or \
after it:

---
title: <short descriptive title>
citation: <author/venue/year, e.g. "Rendle 2010, ICDM">
tags: <comma-separated keywords useful for later retrieval>
source_url: <the single most authoritative URL backing this note's core claim>
---
<4-8 sentences: the core idea, when it helps, its cost/complexity, and anything specific to a \
KuaiRand-Pure-style sparse, popularity-skewed short-video ranking task, if relevant. Every \
non-obvious factual claim must be traceable to source_url above — if you drew on more than one \
source, pick the single best one for source_url and name the others by publisher in the body text \
instead of adding more source_url fields.>

If you genuinely found nothing trustworthy, respond with EXACTLY this and nothing else:
NO_RELIABLE_SOURCE_FOUND: <one sentence on what you tried>"""


def build_research_prompt(question):
    return f"""Question: {question}

Search the web now and answer using the fixed note format from your instructions."""


_NOTE_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL | re.MULTILINE)
_URL_HOST_RE = re.compile(r'https?://([^/\s]+)')


def _domain_allowed(url):
    m = _URL_HOST_RE.match(url.strip())
    if not m:
        return False
    host = m.group(1).lower().split(':')[0]  # strip a port if present
    return any(host == d or host.endswith('.' + d) for d in ALLOWED_DOMAINS)


def parse_research_response(text):
    """Returns None if no reliable source was found, the response didn't parse, or the
    cited domain isn't in ALLOWED_DOMAINS (checked here in code -- never trusted purely
    on the model's say-so). Otherwise returns {'title', 'citation', 'tags', 'source_url',
    'body'}."""
    text = text.strip()
    if not text or text.startswith('NO_RELIABLE_SOURCE_FOUND'):
        return None
    m = _NOTE_RE.search(text)  # search, not match -- a stray preamble sentence before the
                                # frontmatter would otherwise make a valid note unparseable
    if not m:
        return None
    header, body = m.group(1), m.group(2).strip()
    meta = {}
    for line in header.strip().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    source_url = meta.get('source_url', '')
    if not source_url or not _domain_allowed(source_url):
        return None
    if not body:
        return None
    return {'title': meta.get('title', 'untitled') or 'untitled', 'citation': meta.get('citation', ''),
            'tags': meta.get('tags', ''), 'source_url': source_url, 'body': body}


def _slug(title):
    s = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
    return s[:60] or 'note'


def save_note(note, node_id):
    """`source: web` must be exactly that string, not a compound one -- rag.py's
    build_grounding_context() does an exact-match check (`!= 'web'`) to decide whether
    to label a retrieved note '[curated]' or '[found via live web search this run]', so
    a value like 'web (found at node 3)' would silently fail that check and mislabel
    every web-found note as curated. The node id still goes on disk, just as its own
    field."""
    WEB_LITERATURE_DIR.mkdir(parents=True, exist_ok=True)
    path = WEB_LITERATURE_DIR / f"{node_id}_{_slug(note['title'])}.md"
    text = (f"---\ntitle: {note['title']}\ncitation: {note['citation']}\n"
            f"tags: {note['tags']}\nsource_url: {note['source_url']}\n"
            f"source: web\nfound_at_node: {node_id}\n---\n{note['body']}\n")
    path.write_text(text, encoding='utf-8')
    return path


def run_research(question, node_id, model=llm.DEFAULT_MODEL, max_budget_usd=llm.DEFAULT_MAX_BUDGET_USD,
                  timeout=llm.DEFAULT_RESEARCH_TIMEOUT_S):
    """One research call. Never raises -- a rejected/failed result just means this node
    spent its research-budget slot and found nothing usable, same as a failed probe.
    Returns {'ok': bool, 'question': str, 'note': dict|None, 'rejected_reason': str|None,
    'cost_usd': float, 'saved_path': str|None}."""
    prompt = build_research_prompt(question)
    result = llm.call_claude_research(SYSTEM_PROMPT, prompt, model=model,
                                       max_budget_usd=max_budget_usd, timeout=timeout)
    if not result['ok']:
        return {'ok': False, 'question': question, 'note': None,
                'rejected_reason': f"LLM call failed: {result['error']}",
                'cost_usd': 0.0, 'saved_path': None}

    note = parse_research_response(result['text'])
    if note is None:
        return {'ok': False, 'question': question, 'note': None,
                'rejected_reason': 'no note with an allowlisted citation was returned',
                'cost_usd': result.get('cost_usd', 0.0), 'saved_path': None}

    path = save_note(note, node_id)
    return {'ok': True, 'question': question, 'note': note, 'rejected_reason': None,
            'cost_usd': result.get('cost_usd', 0.0), 'saved_path': str(path)}
