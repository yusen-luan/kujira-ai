"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine，pairwise BPR loss（同用户内 正/负样本对比），
                  optionally bagged over n_bag independently-seeded models (variance reduction)
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity（官方 baseline） ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine (pairwise BPR training) ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _adam_update(self, gV, gW):
        gV = gV + self.l2 * self.V
        gW = gW + self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def bpr_step(self, Xpos, Xneg):
        """Pairwise BPR update: push score(pos) above score(neg) for same-user pairs."""
        B = len(Xpos)
        zpos, Epos, Spos = self.logits(Xpos)
        zneg, Eneg, Sneg = self.logits(Xneg)
        diff = zpos - zneg
        p = sigmoid(diff)
        g = ((p - 1.0) / B).astype(np.float32)   # dL/dzpos = g ; dL/dzneg = -g

        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xpos, g[:, None])
        np.add.at(gW, Xneg, -g[:, None])
        np.add.at(gV, Xpos, g[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, -g[:, None, None] * (Sneg[:, None, :] - Eneg))
        self._adam_update(gV, gW)
        # bias cancels out in the pairwise difference (contributes equally to both sides)
        return float(-np.mean(np.log(p + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def _build_pair_index(y, users, neg_ratio, seed):
    """For each positive (label=1) train row, find candidate negative (label=0) rows
    from the SAME user. Returns arrays enabling vectorized per-epoch resampling of
    negatives (row indices are into the original train X/y arrays)."""
    y = np.asarray(y)
    users_arr = np.asarray(users)
    uniq, uinv = np.unique(users_arr, return_inverse=True)
    n_users = len(uniq)

    pos_mask = y > 0.5
    neg_mask = ~pos_mask

    neg_users = uinv[neg_mask]
    order = np.argsort(neg_users, kind='stable')
    neg_rows_all = np.nonzero(neg_mask)[0][order]
    neg_users_sorted = neg_users[order]
    counts = np.bincount(neg_users_sorted, minlength=n_users).astype(np.int64)
    offsets = np.zeros(n_users, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)[:-1]

    pos_rows = np.nonzero(pos_mask)[0]
    pos_users = uinv[pos_mask]
    keep = counts[pos_users] > 0
    pos_rows = pos_rows[keep]
    pos_users = pos_users[keep]

    if neg_ratio > 1:
        pos_rows = np.tile(pos_rows, neg_ratio)
        pos_users = np.tile(pos_users, neg_ratio)

    return pos_rows, pos_users, neg_rows_all, offsets, counts


def _train_one_fm(Xtr, pos_rows, pos_users, neg_rows_all, neg_offsets, neg_counts,
                   dim, k, lr, l2, epochs, bs, patience, seed, verbose, tag):
    """Train a single FM via BPR to its own early-stopped best state; return it."""
    m = FM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    n_pairs = len(pos_rows)

    best, best_state, bad = -1, (m.V.copy(), m.W.copy(), np.float32(m.b)), 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        r = rng.integers(0, neg_counts[pos_users])
        neg_row_idx = neg_rows_all[neg_offsets[pos_users] + r]
        perm = rng.permutation(n_pairs)

        losses = []
        for i in range(0, n_pairs, bs):
            b_idx = perm[i:i + bs]
            Xpos = Xtr[pos_rows[b_idx]]
            Xneg = Xtr[neg_row_idx[b_idx]]
            losses.append(m.bpr_step(Xpos, Xneg))

        va = _EVAL_HOOK[0](m)
        if verbose:
            print(f"  [{tag}] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  [{tag}] early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m

# module-level hook used inside _train_one_fm to avoid threading eval-closure args everywhere
_EVAL_HOOK = [None]


def run_fm(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
           neg_per_pos=1, n_bag=1):
    """Train n_bag independently-seeded FMs on the same BPR objective and average their
    raw logits at prediction time (bagging). n_bag=1 reproduces the original single-model
    behavior exactly."""
    neg_per_pos = max(1, int(round(neg_per_pos)))
    n_bag = max(1, int(round(n_bag)))
    enc, dim = encode(splits)
    Xtr, ytr, users_tr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    pos_rows, pos_users, neg_rows_all, neg_offsets, neg_counts = _build_pair_index(
        ytr, users_tr, neg_per_pos, seed)

    _EVAL_HOOK[0] = lambda m: evaluate(uva, yva, m.predict(Xva))

    va_preds, te_preds = [], []
    for bag_i in range(n_bag):
        bag_seed = int(seed) + bag_i * 997
        tag = f"bag{bag_i}" if n_bag > 1 else "fm"
        m = _train_one_fm(Xtr, pos_rows, pos_users, neg_rows_all, neg_offsets, neg_counts,
                           dim, k, lr, l2, epochs, bs, patience, bag_seed, verbose, tag)
        va_preds.append(m.predict(Xva))
        te_preds.append(m.predict(Xte))

    va_avg = np.mean(va_preds, axis=0)
    te_avg = np.mean(te_preds, axis=0)
    return {'valid': evaluate(uva, yva, va_avg),
            'test':  evaluate(ute, yte, te_avg)}

def run_model(splits, hparams, seed=0, verbose=True):
    """Generic entrypoint the harness (run_and_report.py) always calls, regardless
    of model family. Reads only the keys it needs via hparams.get(...) with sensible
    defaults, so an unrelated/missing key in the generic hparams dict never raises."""
    return run_fm(
        splits,
        k=hparams.get('k', 16),
        lr=hparams.get('lr', 0.001),
        l2=hparams.get('l2', 1e-6),
        epochs=hparams.get('epochs', 40),
        bs=hparams.get('bs', 8192),
        patience=hparams.get('patience', 4),
        neg_per_pos=hparams.get('neg_per_pos', 1),
        n_bag=hparams.get('n_bag', 1),
        seed=seed,
        verbose=verbose,
    )

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--neg_per_pos', type=int, default=1)
    ap.add_argument('--n_bag', type=int, default=1)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, l2=a.l2, epochs=a.epochs, seed=a.seed,
                                   neg_per_pos=a.neg_per_pos, n_bag=a.n_bag)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")