"""Fixed execution harness for one LLM-authored diagnostic probe — orchestrator-owned,
the LLM never sees or edits this file (same posture as run_and_report.py for the main
data.py/baseline.py axes).

A probe never touches the filesystem itself: it receives the exact same pre-loaded
objects agent/eda.py's own build_report() computes internally (arrays, masks,
side-info dicts) as plain function arguments, reusing eda.py's loaders directly. This
means a probe has no filesystem-access surface beyond what eda.py already reads, and
no way to see anything eda.py couldn't — it only gets to ask a different question of
the same data.

Contract the orchestrator relies on (mirrors run_and_report.py):
  exit code 0 and a valid --out JSON file  -> success
  non-zero exit (traceback on stderr), or timeout, or missing/invalid --out
                                            -> failure, retry/rollback

probe.py must define `run_probe(arrays, masks, user_feat, video_feat, label) -> dict`
(JSON-serializable). See prompt_templates.PROBE_SYSTEM_PROMPT for the exact argument
shapes handed to it.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
import eda  # noqa: E402  reuse the same loaders build_report() uses, so a probe sees
            # exactly what eda.py sees, computed the exact same way


def to_native(obj):
    """Recursively convert numpy scalars to plain Python types for json.dump."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if hasattr(obj, 'item'):  # numpy scalar (float32, int64, ...)
        return obj.item()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe_dir', required=True, help='dir containing the candidate probe.py')
    ap.add_argument('--data_dir', required=True, help='KuaiRand-Pure data dir')
    ap.add_argument('--out', required=True, help='path to write the probe result JSON to')
    a = ap.parse_args()

    print('  loading pre-computed EDA arrays (same loaders as agent/eda.py)...', flush=True)
    t0 = time.time()
    arrays = eda.load_logs_as_arrays(a.data_dir)
    masks = eda.split_masks(arrays['date'])
    user_feat = eda.read_side(os.path.join(a.data_dir, 'user_features_pure.csv'), 'user_id')
    video_feat = eda.read_side(os.path.join(a.data_dir, 'video_features_basic_pure.csv'), 'video_id')
    print(f'  loaded ({time.time() - t0:.0f}s), running probe.run_probe()...', flush=True)

    sys.path.insert(0, a.probe_dir)
    import probe  # the LLM-authored file, applied into a.probe_dir by orchestrator.py

    result = probe.run_probe(arrays, masks, user_feat, video_feat, eda.LABEL)
    if not isinstance(result, dict):
        raise TypeError(f'run_probe must return a dict, got {type(result).__name__}')

    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(to_native(result), fh, indent=2)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
