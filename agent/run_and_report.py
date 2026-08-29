"""Fixed execution harness — orchestrator-owned, the LLM never sees or edits this file.

Imports a candidate's data.py + baseline.py as plain modules, runs the candidate's
model, and writes {'valid': {...}, 'test': {...}} to --out as JSON.

Contract the orchestrator relies on:
  exit code 0 and a valid --out JSON file  -> success
  non-zero exit (traceback on stderr), or timeout, or missing/invalid --out
                                            -> failure, retry/rollback

Model-family-agnostic on purpose: baseline.py must expose
`run_model(splits, hparams: dict, seed=0, verbose=True) -> {'valid':..., 'test':...}`.
hparams is an opaque dict (FM's {k, lr, epochs, bs, patience}, or a different
model family's own hyperparameter names) so this harness doesn't need to change
when the candidate swaps model families.
"""
import argparse
import json
import sys
import time
import traceback


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
    ap.add_argument('--candidate_dir', required=True, help='dir containing the candidate data.py + baseline.py')
    ap.add_argument('--pinned_dir', required=True, help='dir containing the pinned evaluate.py (not copied/editable)')
    ap.add_argument('--data_dir', required=True, help='KuaiRand-Pure data dir (shared, never copied per-iteration)')
    ap.add_argument('--out', required=True, help='path to write metrics JSON to')
    ap.add_argument('--hparams', required=True, help='JSON dict passed to baseline.run_model')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verbose', action='store_true',
                     help='print data-loading status and pass verbose=True to run_model '
                          '(per-epoch training progress, if the candidate prints it)')
    a = ap.parse_args()
    hparams = json.loads(a.hparams)

    # candidate_dir first so its data.py/baseline.py shadow anything of the same
    # name; pinned_dir after, so `import evaluate` still resolves to the real,
    # never-copied evaluate.py that baseline.py imports internally.
    sys.path.insert(0, a.candidate_dir)
    sys.path.insert(1, a.pinned_dir)
    import data
    import baseline

    if a.verbose:
        print(f'  loading data from {a.data_dir} ...', flush=True)
    t0 = time.time()
    splits = data.load(a.data_dir)
    if a.verbose:
        counts = ', '.join(f'{k}={len(v)}' for k, v in splits.items())
        print(f'  loaded ({time.time() - t0:.0f}s): {counts}', flush=True)
        print('  starting training...', flush=True)
    result = baseline.run_model(splits, hparams=hparams, seed=a.seed, verbose=a.verbose)

    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(to_native(result), fh, indent=2)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
