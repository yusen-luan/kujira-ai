"""KuaiRand-Pure baselines。
  --model pop    : item popularity（官方 baseline，纯统计，不训练）
  --model fm     : Factorization Machine，pointwise BCE（原始起步模型）
  --model bpr    : FM + BPR pairwise（numpy 实现，历史最优 order-2 baseline）
  --model deepfm : FM + MLP (DeepFM-style), BPR pairwise（纯 numpy 实现，当前 run_model 默认）
  --model random : 随机打分（下界，用来自检评测代码没坏）
默认 run_model 走 DeepFM + BPR pairwise 训练流程（见 run_deepfm_bpr）。
NOTE: torch is unavailable in this runtime, so the DeepFM-style extra-capacity
test is implemented as a pure-numpy MLP-on-top-of-shared-FM-embeddings model
(manual forward/backward), trained with the exact same BPR pairwise loss and
same-user negative-sampling scheme as run_fm_bpr, so this is still a clean test
of "does adding an MLP branch for higher-order interactions help" without
introducing any other confound. Everything stays numpy + stdlib only, matching
the hard constraint that torch may be used only if actually installed/available.
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

# ---------------- Factorization Machine (numpy, shared core) ----------------
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

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        self._adam_apply(gV, gW, float(g.sum()))
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

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


def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
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
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    ytr_i = ytr.astype(np.int64)
    user_col = Xtr[:, 0].astype(np.int64)

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

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0

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
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


# ---------------- DeepFM-lite (pure numpy): FM order-2 term + MLP over
# concatenated field embeddings (shared embeddings), trained with BPR ----------------
class DeepFMNP:
    """FM(order-2, shared embeddings) + MLP(flattened embeddings) -> scalar.
    Manual forward/backward in numpy so it has no torch dependency."""

    def __init__(self, dim, num_fields, k=16, mlp_dims=(64, 32), lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2

        dims = [num_fields * k] + list(mlp_dims) + [1]
        self.Ws = [rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32)
                   for i in range(len(dims) - 1)]
        self.bs = [np.zeros(dims[i + 1], dtype=np.float32) for i in range(len(dims) - 1)]

        # adam moment buffers
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.float32(0.0); self.vb = np.float32(0.0)
        self.mWs = [np.zeros_like(w) for w in self.Ws]; self.vWs = [np.zeros_like(w) for w in self.Ws]
        self.mbs = [np.zeros_like(b) for b in self.bs]; self.vbs = [np.zeros_like(b) for b in self.bs]
        self.t = 0

    def forward(self, X):
        E = self.V[X]                                # (B,F,k)
        S = E.sum(1)                                  # (B,k)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.W[X].sum(1)
        flat = E.reshape(E.shape[0], -1)
        acts = [flat]
        pre_acts = []
        h = flat
        for i in range(len(self.Ws) - 1):
            pre = h @ self.Ws[i] + self.bs[i]
            pre_acts.append(pre)
            h = np.maximum(pre, 0)
            acts.append(h)
        pre_last = h @ self.Ws[-1] + self.bs[-1]
        mlp_out = pre_last.squeeze(-1)
        z = self.b + lin + fm_inter + mlp_out
        cache = (X, E, S, acts, pre_acts)
        return z.astype(np.float32), cache

    def backward(self, cache, dz):
        X, E, S, acts, pre_acts = cache
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        gb = float(dz.sum())

        np.add.at(gW, X, dz[:, None])
        np.add.at(gV, X, dz[:, None, None] * (S[:, None, :] - E))

        gWs = [None] * len(self.Ws)
        gbs = [None] * len(self.bs)

        dh_last = dz[:, None]                         # (B,1), d mlp_out / d pre_last = 1
        h_prev = acts[-1]
        gWs[-1] = h_prev.T @ dh_last
        gbs[-1] = dh_last.sum(0)
        dh = dh_last @ self.Ws[-1].T                  # (B, hidden_last)

        for i in range(len(self.Ws) - 2, -1, -1):
            pre = pre_acts[i]
            relu_mask = (pre > 0).astype(dz.dtype)
            dpre = dh * relu_mask
            h_in = acts[i]
            gWs[i] = h_in.T @ dpre
            gbs[i] = dpre.sum(0)
            dh = dpre @ self.Ws[i].T

        dflat = dh                                    # (B, F*k) == dL/dflat
        dflat_reshaped = dflat.reshape(E.shape)
        np.add.at(gV, X, dflat_reshaped)

        return gV, gW, gb, gWs, gbs

    def _adam_1(self, P, G, M, Vv):
        b1, b2, eps = 0.9, 0.999, 1e-8
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def apply_grads(self, gV, gW, gb, gWs, gbs):
        gV = gV + self.l2 * self.V
        gW = gW + self.l2 * self.W
        self.t += 1
        self._adam_1(self.V, gV, self.mV, self.vV)
        self._adam_1(self.W, gW, self.mW, self.vW)
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.mb = b1 * self.mb + (1 - b1) * gb
        self.vb = b2 * self.vb + (1 - b2) * (gb * gb)
        self.b -= self.lr * (self.mb / (1 - b1 ** self.t)) / (np.sqrt(self.vb / (1 - b2 ** self.t)) + eps)
        for i in range(len(self.Ws)):
            gWi = gWs[i] + self.l2 * self.Ws[i]
            self._adam_1(self.Ws[i], gWi, self.mWs[i], self.vWs[i])
            self._adam_1(self.bs[i], gbs[i], self.mbs[i], self.vbs[i])

    def step_bpr(self, Xpos, Xneg):
        B = len(Xpos)
        zpos, cache_pos = self.forward(Xpos)
        zneg, cache_neg = self.forward(Xneg)
        d = zpos - zneg
        p = sigmoid(d)
        coef = ((p - 1.0) / B).astype(np.float32)      # dL/dz_pos ; dL/dz_neg = -coef

        gVp, gWp, gbp, gWsp, gbsp = self.backward(cache_pos, coef)
        gVn, gWn, gbn, gWsn, gbsn = self.backward(cache_neg, -coef)

        gV = gVp + gVn
        gW = gWp + gWn
        gb = gbp + gbn
        gWs = [a + b for a, b in zip(gWsp, gWsn)]
        gbs = [a + b for a, b in zip(gbsp, gbsn)]
        self.apply_grads(gV, gW, gb, gWs, gbs)
        return float(np.mean(-np.log(p + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.forward(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def state(self):
        return (self.V.copy(), self.W.copy(), float(self.b),
                [w.copy() for w in self.Ws], [b.copy() for b in self.bs])

    def load_state(self, st):
        self.V, self.W, b, Ws, bs = st
        self.b = np.float32(b)
        self.Ws = [w.copy() for w in Ws]
        self.bs = [b_.copy() for b_ in bs]


def run_deepfm_bpr(splits, k=16, mlp_dims=(64, 32), lr=0.001, l2=1e-6,
                    epochs=15, bs=8192, patience=3, neg_k=1, seed=0, verbose=True,
                    **kwargs):
    """DeepFM-lite (FM order-2 term + MLP over concatenated, shared field
    embeddings), trained with the same BPR pairwise loss / same-user negative
    sampling scheme as the previous best (run_fm_bpr). Tests whether the
    plateaued order-2-only FM was capacity-limited. Pure numpy (no torch dep).
    Accepts and ignores any extra hparams (e.g. 'dropout') that don't apply to
    this pure-numpy architecture, since run_model forwards a generic hparams
    dict that may carry keys meant for other model families."""
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    num_fields = Xtr.shape[1]

    if isinstance(mlp_dims, (list, tuple)):
        mlp_dims = tuple(int(x) for x in mlp_dims)
    else:
        mlp_dims = (64, 32)

    ytr_i = ytr.astype(np.int64)
    user_col = Xtr[:, 0].astype(np.int64)

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

    model = DeepFMNP(dim, num_fields, k=k, mlp_dims=mlp_dims, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0

    for ep in range(1, epochs + 1):
        order_pos = rng.permutation(len(pos_row_idx)); t0 = time.time()
        losses = []
        for i in range(0, len(order_pos), bs):
            pos_batch = pos_row_idx[order_pos[i:i + bs]]
            neg_batch = sample_negs(pos_batch, rng)
            for _ in range(max(1, neg_k)):
                if neg_k > 1:
                    neg_batch = sample_negs(pos_batch, rng)
                losses.append(model.step_bpr(Xtr[pos_batch], Xtr[neg_batch]))
        va = evaluate(uva, yva, model.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = model.state()
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    model.load_state(best_state)
    return {'valid': evaluate(uva, yva, model.predict(Xva)),
            'test':  evaluate(ute, yte, model.predict(Xte))}


def run_model(splits, hparams, seed=0, verbose=True):
    """Generic entrypoint the harness (run_and_report.py) always calls, regardless
    of model family. Default: DeepFM-lite (FM + MLP, pure numpy) trained with
    BPR pairwise loss. The harness always passes the same generic hparams dict
    regardless of which model family is active, so we only forward keys that
    run_deepfm_bpr actually understands and silently drop the rest (e.g.
    'bpr', 'neg_pool', 'dropout' which belong to other model families)."""
    kwargs = dict(hparams) if hparams else {}
    allowed = {'k', 'mlp_dims', 'lr', 'l2', 'epochs', 'bs', 'patience', 'neg_k'}
    kwargs = {kk: vv for kk, vv in kwargs.items() if kk in allowed}
    return run_deepfm_bpr(splits, seed=seed, verbose=verbose, **kwargs)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='deepfm', choices=['pop', 'fm', 'bpr', 'deepfm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'bpr': lambda s: run_fm_bpr(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'deepfm': lambda s: run_deepfm_bpr(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")