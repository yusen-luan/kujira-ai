"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine，pointwise BCE（原始起步模型）
  --model bpr   : plain FM + BPR pairwise loss
  --model fwfm  : Field-weighted FM + BPR pairwise loss (current default)
  --model random: 随机打分（下界，用来自检评测代码没坏）
默认 run_model 走 Field-weighted FM 的 BPR pairwise 训练流程（见 run_fwfm_bpr）。
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

# ---------------- Factorization Machine (shared core) ----------------
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

    def _adam_apply(self, gV, gW, gb):
        gV = gV + self.l2 * self.V
        gW = gW + self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * gb

    # ---- pointwise BCE step (original) ----
    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        self._adam_apply(gV, gW, float(g.sum()))
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    # ---- BPR pairwise step ----
    def step_bpr(self, Xpos, Xneg):
        B = len(Xpos)
        zpos, Epos, Spos = self.logits(Xpos)
        zneg, Eneg, Sneg = self.logits(Xneg)
        d = zpos - zneg
        p = sigmoid(d)
        coef = ((p - 1.0) / B).astype(np.float32)         # dL/dz_pos
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xpos, coef[:, None])
        np.add.at(gV, Xpos, coef[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gW, Xneg, (-coef)[:, None])
        np.add.at(gV, Xneg, (-coef)[:, None, None] * (Sneg[:, None, :] - Eneg))
        self._adam_apply(gV, gW, 0.0)   # bias term cancels exactly in a difference
        return float(np.mean(-np.log(p + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


# ---------------- Field-weighted Factorization Machine ----------------
# Per Pan et al., "Field-weighted Factorization Machines for CTR Prediction
# in Display Advertising" (WWW 2018): keep ONE shared embedding table (unlike
# full FFM, which needs a separate embedding per interacting field pair and
# blows up as O(num_fields^2 x k) parameters), but learn a scalar weight
# r_{i,j} per FIELD PAIR that rescales that pair's interaction strength.
# With only 5 fields here that's just 10 extra scalar parameters, initialized
# to 1.0 so training starts out identical to plain FM and only diverges as
# the field-pair weights are learned.
class FwFM:
    def __init__(self, dim, num_fields, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.F = num_fields
        self.iu = np.triu_indices(num_fields, k=1)
        self.P = len(self.iu[0])
        self.r = np.ones(self.P, dtype=np.float32)      # start == plain FM
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mr = np.zeros_like(self.r); self.vr = np.zeros_like(self.r)
        self.t = 0

    def _R(self):
        R = np.zeros((self.F, self.F), dtype=np.float32)
        R[self.iu] = self.r
        R = R + R.T
        return R

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        R = self._R()
        dot_all = np.einsum('bik,bjk->bij', E, E)        # (B,F,F)
        interaction = 0.5 * np.einsum('ij,bij->b', R, dot_all)
        z = self.b + self.W[X].sum(1) + interaction
        return z, E, R, dot_all

    def _adam_apply(self, gV, gW, gr, gb):
        gV = gV + self.l2 * self.V
        gW = gW + self.l2 * self.W
        gr = gr + self.l2 * self.r
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P_, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                              (self.W, gW, self.mW, self.vW),
                              (self.r, gr, self.mr, self.vr)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P_ -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * gb

    def _grads(self, X, E, R, dot_all, coef):
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, coef[:, None])
        gradE = np.einsum('ij,bjk->bik', R, E)            # (B,F,k)
        for f in range(self.F):
            np.add.at(gV, X[:, f], coef[:, None] * gradE[:, f, :])
        gr = np.einsum('b,bij->ij', coef, dot_all)[self.iu]
        return gV, gW, gr

    def step_bpr(self, Xpos, Xneg):
        B = len(Xpos)
        zpos, Epos, R, dot_pos = self.logits(Xpos)
        zneg, Eneg, _, dot_neg = self.logits(Xneg)
        d = zpos - zneg
        p = sigmoid(d)
        coef = ((p - 1.0) / B).astype(np.float32)
        gV_p, gW_p, gr_p = self._grads(Xpos, Epos, R, dot_pos, coef)
        gV_n, gW_n, gr_n = self._grads(Xneg, Eneg, R, dot_neg, -coef)
        self._adam_apply(gV_p + gV_n, gW_p + gW_n, gr_p + gr_n, 0.0)
        return float(np.mean(-np.log(p + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    """Original pointwise-BCE training loop, kept for reference / --model fm CLI use."""
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def run_fm_bpr(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, neg_k=1, seed=0, verbose=True):
    """Plain-FM BPR pairwise-ranking training loop (kept for reference / --model bpr)."""
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    return _bpr_loop(m, Xtr, ytr, Xva, yva, uva, Xte, yte, ute,
                      epochs=epochs, bs=bs, patience=patience, neg_k=neg_k, seed=seed, verbose=verbose)


def run_fwfm_bpr(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, neg_k=1, seed=0, verbose=True):
    """Field-weighted FM (FwFM) + BPR pairwise loss. See FwFM class docstring above:
    learns a scalar interaction-strength weight per field PAIR (10 extra params for
    our 5 fields) instead of full field-aware per-field-pair embeddings, per the
    'field-weighted FM as cheap alternative to full field-awareness' literature note."""
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    num_fields = Xtr.shape[1]
    m = FwFM(dim, num_fields, k=k, lr=lr, seed=seed)
    return _bpr_loop(m, Xtr, ytr, Xva, yva, uva, Xte, yte, ute,
                      epochs=epochs, bs=bs, patience=patience, neg_k=neg_k, seed=seed, verbose=verbose,
                      extra_state=lambda: m.r.copy(), restore_extra=lambda st: setattr(m, 'r', st))


def _bpr_loop(m, Xtr, ytr, Xva, yva, uva, Xte, yte, ute, epochs, bs, patience, neg_k, seed, verbose,
              extra_state=None, restore_extra=None):
    """Shared BPR pairwise training loop (same-user negative sampling, vectorized)
    used by both run_fm_bpr and run_fwfm_bpr, factored out so the only difference
    between the two model families is which `m` object is passed in."""
    ytr_i = ytr.astype(np.int64)
    user_col = Xtr[:, 0].astype(np.int64)                 # field 0 is user_id, offset 0

    pos_row_idx = np.where(ytr_i == 1)[0]
    neg_row_idx = np.where(ytr_i == 0)[0]
    neg_users = user_col[neg_row_idx]

    order = np.argsort(neg_users, kind='stable')
    neg_row_idx_sorted = neg_row_idx[order]
    neg_users_sorted = neg_users[order]
    uniq_u, start_idx, counts = np.unique(neg_users_sorted, return_index=True, return_counts=True)

    max_user = int(user_col.max()) + 1
    neg_start = np.zeros(max_user, dtype=np.int64)
    neg_count = np.zeros(max_user, dtype=np.int64)
    neg_start[uniq_u] = start_idx
    neg_count[uniq_u] = counts
    n_neg_total = len(neg_row_idx_sorted)

    rng = np.random.default_rng(seed)
    best, best_state, best_extra, bad = -1, None, None, 0

    def sample_negs(pos_idx_batch, rng):
        u = user_col[pos_idx_batch]
        u_clamped = np.clip(u, 0, max_user - 1)
        cnt = neg_count[u_clamped]
        has_neg = cnt > 0
        offset = rng.integers(0, np.maximum(cnt, 1))
        neg_pos = neg_start[u_clamped] + offset
        neg_pos = np.clip(neg_pos, 0, n_neg_total - 1)
        chosen = neg_row_idx_sorted[neg_pos]
        n_missing = int((~has_neg).sum())
        if n_missing > 0:
            chosen = chosen.copy()
            chosen[~has_neg] = neg_row_idx_sorted[rng.integers(0, n_neg_total, size=n_missing)]
        return chosen

    for ep in range(1, epochs + 1):
        order_pos = rng.permutation(len(pos_row_idx)); t0 = time.time()
        losses = []
        for i in range(0, len(order_pos), bs):
            pos_batch = pos_row_idx[order_pos[i:i + bs]]
            neg_batch = sample_negs(pos_batch, rng)
            for _ in range(max(1, neg_k)):
                if neg_k > 1:
                    neg_batch = sample_negs(pos_batch, rng)
                losses.append(m.step_bpr(Xtr[pos_batch], Xtr[neg_batch]))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            if extra_state is not None:
                best_extra = extra_state()
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    if restore_extra is not None and best_extra is not None:
        restore_extra(best_extra)
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def run_model(splits, hparams, seed=0, verbose=True):
    """Generic entrypoint the harness (run_and_report.py) always calls, regardless
    of model family. Default: Field-weighted FM + BPR pairwise training (see run_fwfm_bpr)."""
    kwargs = dict(hparams)
    kwargs.pop('bpr', None)
    return run_fwfm_bpr(splits, seed=seed, verbose=verbose, **kwargs)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fwfm', choices=['pop', 'fm', 'bpr', 'fwfm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'bpr': lambda s: run_fm_bpr(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'fwfm': lambda s: run_fwfm_bpr(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")