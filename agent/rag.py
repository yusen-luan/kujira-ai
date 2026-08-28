"""Local retrieval over a small curated literature corpus (agent/literature/*.md).

Pinned, deterministic, no LLM call and no network access at run time. The corpus
itself was curated once, offline, by fetching and verifying real sources (see each
note's `citation` field and AGENT.md) -- nothing here re-fetches anything or gives
the propose-loop LLM any tool/internet access. Retrieval is plain BM25 over term
frequencies (stdlib + numpy only), not embeddings: the corpus is tiny (~9 short
notes) and each note is written with deliberate trigger keywords in its `tags`
field, so simple keyword ranking is enough and avoids a new dependency/API cost.

Query construction is EDA-driven (see build_query): it reads the already-computed
agent/runs/eda_report.json and turns the flagged findings (secondary-signal
sparsity, popularity skew, cold-start rate, data-quality flags) into query terms,
so retrieval favors notes relevant to *this* data rather than a generic search.
Runs once per project (like eda.py), cached to agent/runs/literature_context.md
and reused across a whole orchestrator run.
"""
import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent
LITERATURE_DIR = AGENT_DIR / 'literature'
RUNS_DIR = AGENT_DIR / 'runs'
REPORT_PATH = RUNS_DIR / 'eda_report.json'
CONTEXT_PATH = RUNS_DIR / 'literature_context.md'

sys.path.insert(0, str(REPO_ROOT / 'workspace'))
from data import LABEL  # noqa: E402  -- eda_report.json's LABEL-prefixed keys use this

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _parse_doc(path):
    """Splits a `---\\nkey: val\\n---\\nbody` note into (meta dict, body text)."""
    text = path.read_text(encoding='utf-8')
    meta = {'title': path.stem, 'citation': '', 'tags': ''}
    body = text
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end != -1:
            header, body = text[3:end], text[end + 4:]
            for line in header.strip().splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()
    return {'path': path, 'meta': meta, 'body': body.strip()}


def load_corpus():
    return [_parse_doc(p) for p in sorted(LITERATURE_DIR.glob('*.md'))]


def _bm25_scores(docs, query, k1=1.5, b=0.75):
    doc_tokens = [_tokenize(' '.join([d['meta'].get('tags', ''), d['meta'].get('title', ''), d['body']]))
                  for d in docs]
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in doc_tokens) / n
    df = collections.Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    q_terms = _tokenize(query)
    scores = [0.0] * n
    for i, toks in enumerate(doc_tokens):
        tf = collections.Counter(toks)
        dl = len(toks) or 1
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1)
            scores[i] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return scores


def retrieve(query, top_k=5):
    docs = load_corpus()
    scores = _bm25_scores(docs, query)
    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
    return ranked[:top_k]


def build_query(eda_report):
    """Fixed generic recsys terms (so foundational architecture notes always have
    something to match on) plus terms triggered by specific EDA findings (so the
    situational notes -- multi-task, popularity debiasing, dataset background --
    surface only when the data actually calls for them)."""
    terms = ['CTR prediction', 'ranking', 'feature interaction', 'recommendation', 'embedding', 'KuaiRand']

    rates = eda_report.get('label_base_rates_by_split', {}).get('train', {})
    sparse_signals = [c for c, r in rates.items()
                       if c != 'n_rows' and isinstance(r, (int, float)) and r < 0.03]
    if sparse_signals:
        terms += ['sparse', 'imbalance', 'multi-task', 'auxiliary task', 'sample selection bias', 'seesaw']

    pop = eda_report.get(f'{LABEL}_video_popularity_skew_train', {})
    if pop.get('gini_of_positive_counts_across_videos', 0) > 0.5:
        terms += ['popularity bias', 'popularity skew', 'long tail', 'debias', 'exposure bias', 'gini']

    cold = eda_report.get('cold_start_overlap_vs_train', {})
    if any(v.get('unseen_user_frac', 0) > 0.02 or v.get('unseen_video_frac', 0) > 0.02
           for v in cold.values()):
        terms.append('cold start')

    if eda_report.get('data_quality_flags'):
        terms += ['data quality', 'sentinel value', 'missing value']

    return ' '.join(terms)


def build_grounding_context(eda_report, top_k=5):
    query = build_query(eda_report)
    ranked = retrieve(query, top_k=top_k)
    parts = [f"(retrieved via local BM25 search over agent/literature/, query terms: {query})"]
    for doc, score in ranked:
        title = doc['meta'].get('title', doc['path'].stem)
        citation = doc['meta'].get('citation', '')
        header = f"### {title}" + (f"  ({citation})" if citation else '')
        parts.append(f"{header}\n{doc['body']}")
    return '\n\n'.join(parts)


def run(out_dir=RUNS_DIR, top_k=5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f'{REPORT_PATH} not found -- run agent/eda.py first')
    eda_report = json.loads(REPORT_PATH.read_text(encoding='utf-8'))
    context = build_grounding_context(eda_report, top_k=top_k)
    (out_dir / 'literature_context.md').write_text(context, encoding='utf-8')
    return context


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=str(RUNS_DIR))
    ap.add_argument('--top_k', type=int, default=5)
    a = ap.parse_args()
    ctx = run(a.out_dir, a.top_k)
    print(ctx)
