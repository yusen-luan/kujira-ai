"""One-time deterministic EDA over the KuaiRand-Pure raw CSVs.

Pinned analysis code — NOT LLM-authored, NOT re-run per iteration (same posture as
workspace/evaluate.py: fixed semantics, computed once, trusted as ground truth). It
reads the full raw schema (every log column, plus both side-info files) that
data.py deliberately narrows down to 5 fields, and writes two artifacts:

  agent/runs/eda_report.json   structured numbers (source of truth, always written)
  agent/runs/eda_summary.md    a short markdown summary from ONE LLM call that turns
                                the JSON into decision-oriented bullets (skippable via
                                --skip_llm; falls back to a static note on failure)

prompt_templates.py reads eda_summary.md lazily at prompt-build time (not at import
time) and pastes it into every propose/synthesis user prompt as static, data-grounded
context — so feature/model hypotheses are no longer proposed blind to the real data's
class balance, cardinality, cold-start rate, or known leakage traps.

Usage:
    python agent/eda.py                    # compute + one LLM summarization call
    python agent/eda.py --skip_llm         # compute only, no LLM cost
    python agent/eda.py --regen            # ignored here (orchestrator's flag); this
                                            # script always recomputes when invoked
"""
import argparse
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / 'workspace'
AGENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = AGENT_DIR / 'runs'
REPORT_PATH = RUNS_DIR / 'eda_report.json'
SUMMARY_PATH = RUNS_DIR / 'eda_summary.md'

sys.path.insert(0, str(WORKSPACE))
from data import SPLITS, LABEL  # noqa: E402  single source of truth for date ranges + label
sys.path.insert(0, str(AGENT_DIR))
import llm as llm_mod  # noqa: E402

# Every column in log_standard_*.csv, not just the 5 fields data.py currently keeps.
# All are integer-valued in the raw files (verified against the actual CSVs).
LOG_COLS = ['user_id', 'video_id', 'date', 'hourmin', 'time_ms',
            'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
            'long_view', 'play_time_ms', 'duration_ms', 'profile_stay_time',
            'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab']

FEEDBACK_COLS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
                  'is_hate', 'long_view', 'is_profile_enter']

# Columns recorded concurrently with the impression's outcome -> not known at serving
# time in a real system, so using them as raw same-row input features would leak the
# label. Kept in sync manually with prompt_templates.py's _CONTRACT leakage rule.
LEAKAGE_COLS = ['play_time_ms', 'profile_stay_time', 'comment_stay_time'] + \
               [c for c in FEEDBACK_COLS if c != LABEL]

LOG_FILES = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')


def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def load_logs_as_arrays(data_dir):
    per_col = {c: [] for c in LOG_COLS}
    for fname in LOG_FILES:
        with open(os.path.join(data_dir, fname), newline='') as fh:
            r = csv.reader(fh)
            head = next(r)
            idx = {c: head.index(c) for c in LOG_COLS}
            for row in r:
                for c in LOG_COLS:
                    per_col[c].append(row[idx[c]])
    return {c: np.array(v, dtype=np.int64) for c, v in per_col.items()}


def split_masks(date_arr):
    return {name: (date_arr >= lo) & (date_arr <= hi) for name, (lo, hi) in SPLITS.items()}


def read_side(path, key):
    out = {}
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            out[row[key]] = row
    return out


def label_base_rates(arrays, masks):
    out = {}
    for split, m in masks.items():
        rates = {c: float(arrays[c][m].mean()) for c in FEEDBACK_COLS}
        rates['is_rand'] = float(arrays['is_rand'][m].mean())
        rates['n_rows'] = int(m.sum())
        out[split] = rates
    return out


def user_positive_dist(arrays, masks):
    """Per-user positive-count distribution for the LABEL, on the eval splits (valid/test)
    — the same grouping evaluate.py's GAUC/nDCG operate over."""
    n_users = int(arrays['user_id'].max()) + 1
    out = {}
    for split in ('valid', 'test'):
        m = masks[split]
        u, y = arrays['user_id'][m], arrays[LABEL][m]
        pos = np.bincount(u, weights=y, minlength=n_users)
        tot = np.bincount(u, minlength=n_users)
        seen = tot > 0
        pos, tot = pos[seen], tot[seen]
        out[split] = {
            'n_users_with_impressions': int(seen.sum()),
            'pct_users_zero_positive': float((pos == 0).mean()),
            'pct_users_all_positive': float((pos == tot).mean()),
            'pct_users_gauc_eligible': float(((pos > 0) & (pos < tot)).mean()),
            'median_impressions_per_user': float(np.median(tot)),
            'median_positive_per_user': float(np.median(pos)),
        }
    return out


def popularity_skew(arrays, masks):
    """Video popularity skew on the LABEL, train split — explains how much of the
    pop-baseline's score (primary 0.5715, official numbers) is just popularity."""
    n_videos = int(arrays['video_id'].max()) + 1
    m = masks['train']
    v, y = arrays['video_id'][m], arrays[LABEL][m]
    pos = np.bincount(v, weights=y, minlength=n_videos)
    tot = np.bincount(v, minlength=n_videos)
    pos = pos[tot > 0]
    total_pos = pos.sum()
    n = len(pos)
    top1pct_n = max(1, int(0.01 * n))
    top1pct_share = float(np.sort(pos)[::-1][:top1pct_n].sum() / total_pos) if total_pos else 0.0
    sorted_pos = np.sort(pos)
    cum = np.cumsum(sorted_pos)
    gini = float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n) if cum[-1] > 0 else 0.0
    return {
        'n_distinct_videos_with_impressions': int(n),
        'total_positive_impressions': float(total_pos),
        'top1pct_videos_share_of_positives': top1pct_share,
        'gini_of_positive_counts_across_videos': gini,
    }


def cardinality(arrays, masks, user_feat, video_feat):
    m = masks['train']
    return {
        'train_distinct_user_id': int(np.unique(arrays['user_id'][m]).size),
        'train_distinct_video_id': int(np.unique(arrays['video_id'][m]).size),
        'train_distinct_tab': int(np.unique(arrays['tab'][m]).size),
        'user_features_file_rows': len(user_feat),
        'video_features_file_rows': len(video_feat),
        'distinct_author_id_in_video_features_file':
            len(set(r['author_id'] for r in video_feat.values())),
    }


def cold_start(arrays, masks):
    train_users = np.unique(arrays['user_id'][masks['train']])
    train_videos = np.unique(arrays['video_id'][masks['train']])
    out = {}
    for split in ('valid', 'test'):
        m = masks[split]
        u, v = arrays['user_id'][m], arrays['video_id'][m]
        out[split] = {
            'unseen_user_frac': float((~np.isin(u, train_users)).mean()),
            'unseen_video_frac': float((~np.isin(v, train_videos)).mean()),
        }
    return out


def duration_and_watch_ratio(arrays, masks):
    m = masks['train']
    dur = arrays['duration_ms'][m].astype(np.float64)
    quantiles = {f'p{q}': float(np.percentile(dur, q)) for q in (0, 10, 25, 50, 75, 90, 100)}
    play = arrays['play_time_ms'][m].astype(np.float64)
    ratio = play / np.maximum(dur, 1.0)
    y = arrays[LABEL][m]
    ratio_by_label = {
        f'mean_ratio_{LABEL}_0': float(ratio[y == 0].mean()),
        f'mean_ratio_{LABEL}_1': float(ratio[y == 1].mean()),
        f'median_ratio_{LABEL}_0': float(np.median(ratio[y == 0])),
        f'median_ratio_{LABEL}_1': float(np.median(ratio[y == 1])),
    }
    return {'duration_ms_quantiles_train': quantiles,
            f'play_time_over_duration_ratio_by_{LABEL}': ratio_by_label}


def distribution_shift(arrays, masks):
    """Train window is 4/08-4/21; valid/test window is 4/22-5/08 — a date split, not a
    random one, so it's worth checking whether basic distributions actually moved."""
    out = {}
    for split in ('train', 'valid', 'test'):
        m = masks[split]
        tabs = arrays['tab'][m]
        vals, counts = np.unique(tabs, return_counts=True)
        out[split] = {
            'tab_share': {int(v): float(c / m.sum()) for v, c in zip(vals, counts)},
            'mean_duration_ms': float(arrays['duration_ms'][m].mean()),
            f'{LABEL}_rate': float(arrays[LABEL][m].mean()),
        }
    return out


def scan_side_quality(side_dict, fname, id_col):
    """Generic sentinel/bad-value scan: for every mostly-numeric column, flag negative
    values in anything that looks like a non-negative count or 0/1 flag by name. This
    is what catches things like user_features_pure.csv's is_live_streamer containing
    -124 for some rows -- a sentinel code, not a real value -- without hardcoding the
    specific column or value in advance."""
    flags = []
    if not side_dict:
        return flags
    cols = [c for c in next(iter(side_dict.values())).keys() if c != id_col]
    for col in cols:
        numeric = []
        for row in side_dict.values():
            try:
                numeric.append(float(row[col]))
            except ValueError:
                pass
        if len(numeric) < 0.9 * len(side_dict):
            continue  # mostly non-numeric (categorical string like a *_range bucket) -> skip
        arr = np.array(numeric)
        looks_nonnegative = col.startswith('is_') or 'num' in col or 'days' in col
        if looks_nonnegative and arr.min() < 0:
            flags.append({
                'file': fname, 'column': col,
                'issue': f'{int((arr < 0).sum())} row(s) have a negative value in what looks '
                         f'like a non-negative count/flag column -- likely a sentinel/'
                         f'missing-value code, not a real value',
                'example_values': sorted(set(v for v in numeric if v < 0))[:5],
            })
    return flags


def build_report(data_dir):
    arrays = load_logs_as_arrays(data_dir)
    masks = split_masks(arrays['date'])
    user_feat = read_side(os.path.join(data_dir, 'user_features_pure.csv'), 'user_id')
    video_feat = read_side(os.path.join(data_dir, 'video_features_basic_pure.csv'), 'video_id')

    report = {
        'label_used_by_pipeline': LABEL,
        'row_counts': {s: int(masks[s].sum()) for s in SPLITS},
        'label_base_rates_by_split': label_base_rates(arrays, masks),
        f'{LABEL}_per_user_distribution_eval_splits': user_positive_dist(arrays, masks),
        f'{LABEL}_video_popularity_skew_train': popularity_skew(arrays, masks),
        'cardinality': cardinality(arrays, masks, user_feat, video_feat),
        'cold_start_overlap_vs_train': cold_start(arrays, masks),
        **duration_and_watch_ratio(arrays, masks),
        'distribution_shift_by_split': distribution_shift(arrays, masks),
        'data_quality_flags': (
            scan_side_quality(user_feat, 'user_features_pure.csv', 'user_id')
            + scan_side_quality(video_feat, 'video_features_basic_pure.csv', 'video_id')
        ),
        'side_files_available_but_not_yet_used_by_data_py': {
            'user_features_pure.csv': {
                'rows': len(user_feat),
                'columns': list(next(iter(user_feat.values())).keys()) if user_feat else [],
            },
            'video_features_basic_pure.csv': {
                'rows': len(video_feat),
                'columns': list(next(iter(video_feat.values())).keys()) if video_feat else [],
            },
        },
        'leakage_columns_excluded_from_raw_features': LEAKAGE_COLS,
    }
    return to_native(report)


EDA_INTERPRET_SYSTEM_PROMPT = """You turn a structured JSON EDA report about a recommendation \
dataset (KuaiRand-Pure, an engagement-prediction ranking task) into a short, decision-oriented \
markdown summary. This summary gets pasted into every future prompt an ML-engineering LLM agent \
receives before it proposes a feature-engineering or model-architecture change -- so it must be \
concise (tight bullet points, no filler, no restating numbers that aren't actionable) and focused \
on facts that should change what the agent tries next. Organize under these exact headers, in \
this order: `## Label & class balance`, `## Popularity & cold-start`, `## Data quality issues`, \
`## Leakage risk`, `## Distribution shift`, `## Unused data available`, `## Agent's own follow-up \
investigations`. Under Leakage risk, explicitly list which raw log columns must never be used as \
same-row input features and state why in one sentence. Under Unused data available, name the \
side-info files/columns the pipeline doesn't currently read and one concrete way each could be \
used. Under Agent's own follow-up investigations, distill any findings given to you below from \
the agentic-EDA loop's own probes into a few bullets (or write exactly "(none this run)" if none \
were given). Keep the whole thing under ~50 lines of markdown. Output only the markdown, nothing \
else."""


def summarize_with_llm(report, model, max_budget_usd, extra_context=''):
    extra_block = (f"\n\nAdditional findings from the agent's own follow-up investigations "
                    f"(across every agentic-EDA round run so far this project):\n{extra_context}\n"
                    if extra_context else '')
    user_prompt = ("EDA report (JSON):\n```json\n" + json.dumps(report, indent=2) +
                   "\n```" + extra_block + "\n\nWrite the summary now.")
    attempt = llm_mod.call_claude(EDA_INTERPRET_SYSTEM_PROMPT, user_prompt,
                                   model=model, max_budget_usd=max_budget_usd)
    if not attempt['ok']:
        return (f"(EDA summary generation failed: {attempt['error']} -- "
                 f"see agent/runs/eda_report.json for the raw numbers instead.)")
    return attempt['text'].strip()


def run(data_dir, out_dir=RUNS_DIR, model='sonnet', max_budget_usd=0.50, skip_llm=False,
        agent_turns=3, agent_max_budget_usd=1.00, agent_max_repairs=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('computing EDA report (one-time, deterministic)...')
    t0 = time.time()
    report = build_report(data_dir)
    (out_dir / 'eda_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'  wrote {out_dir / "eda_report.json"}  ({time.time() - t0:.0f}s)')

    extra_context = ''
    if agent_turns > 0 and not skip_llm:
        # Lazy: eda_agent.py imports this module (eda) at ITS top level (for the
        # LOG_COLS/FEEDBACK_COLS/LEAKAGE_COLS schema text), so importing it eagerly at
        # this file's top level would be a real load-time cycle. By the time this line
        # runs, eda.py's own module object is already fully initialized in sys.modules,
        # so this is safe -- see probe_runner.py's docstring for why the *other* half
        # (the orchestrator-shared probe-execution helpers) had to move to a real leaf
        # module instead of relying on import-order timing like this one does.
        import eda_agent
        print(f'  running agentic-EDA exploration (up to {agent_turns} turns, '
              f'${agent_max_budget_usd:.2f} cap)...')
        t0 = time.time()
        eda_agent.run_agentic_exploration(
            report, data_dir, model=model, max_turns=agent_turns,
            max_budget_usd=agent_max_budget_usd, max_repairs=agent_max_repairs,
            propose_timeout=300, probe_timeout=90, out_dir=out_dir)
        extra_context = eda_agent.accumulated_findings_text(out_dir)
        print(f'  agentic-EDA exploration done  ({time.time() - t0:.0f}s)')

    if skip_llm:
        summary = '(LLM summary skipped via --skip_llm; see eda_report.json for raw numbers.)'
    else:
        print('  summarizing via one LLM call...')
        t0 = time.time()
        summary = summarize_with_llm(report, model, max_budget_usd, extra_context=extra_context)
        print(f'  summarized  ({time.time() - t0:.0f}s)')
    (out_dir / 'eda_summary.md').write_text(summary, encoding='utf-8')
    print(f'  wrote {out_dir / "eda_summary.md"}')
    return report, summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(WORKSPACE / 'data'))
    ap.add_argument('--out_dir', default=str(RUNS_DIR))
    ap.add_argument('--model', default='sonnet')
    ap.add_argument('--max_budget_usd', type=float, default=0.50)
    ap.add_argument('--skip_llm', action='store_true')
    ap.add_argument('--eda_agent_turns', type=int, default=3,
                     help='max agentic-EDA exploration turns before finalizing the '
                          'summary; 0 disables the loop entirely')
    ap.add_argument('--eda_agent_max_budget_usd', type=float, default=1.00)
    a = ap.parse_args()
    run(a.data_dir, a.out_dir, a.model, a.max_budget_usd, a.skip_llm,
        agent_turns=a.eda_agent_turns, agent_max_budget_usd=a.eda_agent_max_budget_usd)
