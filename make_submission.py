"""Generate the final submission CSV from the agent's current best checkpoint.

Retrains `agent/runs/best/{data.py,baseline.py}` deterministically (fixed seed,
fixed hyperparameters -- the same ones recorded in the corresponding
`logs/node_N.json`), scores every row of the requested split, and writes it via
`workspace/submit.py`'s writer so the output already matches the organizer's
required format (row_id,user_id,video_id,score).

This intentionally does NOT go through `agent/run_and_report.py` (that harness's
`run_model(...) -> {'valid':..., 'test':...}` contract is relied on by the
orchestrator and must stay JSON-only) -- it calls `baseline.run_model(...,
_capture=...)` instead, an opt-in side channel that hands back the trained model
object so its raw per-row scores can be written out, without changing what the
orchestrator itself ever sees.

Usage:
    python3 make_submission.py                # submission/submission.csv, test split
    python3 make_submission.py --split valid  # sanity-check split (labeled, locally scorable)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BEST_DIR = ROOT / 'agent' / 'runs' / 'best'
WORKSPACE = ROOT / 'workspace'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--best_dir', default=str(DEFAULT_BEST_DIR),
                     help='dir containing the candidate data.py + baseline.py to score')
    ap.add_argument('--data_dir', default=str(WORKSPACE / 'data'))
    ap.add_argument('--out', default=str(ROOT / 'submission' / 'submission.csv'))
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    # best_dir first so its data.py/baseline.py shadow anything of the same name;
    # workspace after, so `import evaluate` (inside baseline.py) and `from submit
    # import write_submission` still resolve to the pinned originals -- same
    # sys.path convention as agent/run_and_report.py.
    sys.path.insert(0, a.best_dir)
    sys.path.insert(1, str(WORKSPACE))
    import data
    import baseline
    from submit import write_submission

    print(f'loading data from {a.data_dir} ...')
    splits = data.load(a.data_dir)

    capture = {}
    hparams = {'k': a.k, 'lr': a.lr, 'epochs': a.epochs}
    metrics = baseline.run_model(splits, hparams=hparams, seed=a.seed, verbose=True, _capture=capture)
    print(f"reproduced  valid primary {metrics['valid']['primary']:.4f}  |  "
          f"test primary {metrics['test']['primary']:.4f}")
    print('compare against the accepted node in logs/ to confirm this matches.')

    model = capture['model']
    enc, _dim = data.encode(splits)
    X, _y, _u = enc[a.split]
    scores = model.predict(X)

    write_submission(a.out, splits[a.split], scores)
    print(f'wrote {a.out}: {len(splits[a.split]):,d} rows (split={a.split})')


if __name__ == '__main__':
    main()
