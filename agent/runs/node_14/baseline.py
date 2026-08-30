"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine，pointwise BCE（原始起步模型）
  --model bpr   : BPR pairwise 单任务训练
  --model mt    : BPR + 共享嵌入的辅助任务 (is_like) 联合训练（默认，见 run_fm_bpr_mt）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy（+ torch 允许但当前未使用）。用法见 README.md
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

# ---------------- Factorization Machine (single-task, shared core) ----------------
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


# ---------------- Multi-task FM: shared V, separate linear/bias per task ----------------
class FM_MT:
    """Shared embedding table V feeds two heads: a main head (long_view, trained via BPR
    pairwise loss) and an auxiliary head (is_like, trained via pointwise BCE, weighted by
    alpha). Both heads reuse the SAME quadratic interaction term computed from V, so the
    auxiliary loss directly regularizes the shared embeddings used for main-task ranking."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W_main = np.zeros(dim, dtype=np.float32)
        self.W_aux = np.zeros(dim, dtype=np.float32)
        self.b_main = np.float32(0.0)
        self.b_aux = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mWm = np.zeros_like(self.W_main); self.vWm = np.zeros_like(self.W_main)
        self.mWa = np.zeros_like(self.W_aux); self.vWa = np.zeros_like(self.W_aux)
        self.mba = 0.0; self.vba = 0.0
        self.t = 0

    def _stats(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return E, S, inter

    def main_logits(self, X, inter=None):
        if inter is None:
            _, _, inter = self._stats(X)
        return self.b_main + self.W_main[X].sum(1) + inter

    def step_bpr_mt(self, Xpos, Xneg, aux_pos, aux_neg, alpha=0.1):
        B = len(Xpos)
        Epos, Spos, interpos = self._stats(Xpos)
        Eneg, Sneg, interneg = self._stats(Xneg)

        main_pos = self.b_main + self.W_main[Xpos].sum(1) + interpos
        main_neg = self.b_main + self.W_main[Xneg].sum(1) + interneg
        p = sigmoid(main_pos - main_neg)
        main_coef_pos = ((p - 1.0) / B).astype(np.float32)
        main_coef_neg = -main_coef_pos

        aux_pos_logit = self.b_aux + self.W_aux[Xpos].sum(1) + interpos
        aux_neg_logit = self.b_aux + self.W_aux[Xneg].sum(1) + interneg
        pa_pos = sigmoid(aux_pos_logit)
        pa_neg = sigmoid(aux_neg_logit)
        aux_coef_pos = (alpha * (pa_pos - aux_pos) / B).astype(np.float32)
        aux_coef_neg = (alpha * (pa_neg - aux_neg) / B).astype(np.float32)

        gV = np.zeros_like(self.V)
        gWm = np.zeros_like(self.W_main)
        gWa = np.zeros_like(self.W_aux)

        # main task (BPR) contributions to shared V and W_main
        np.add.at(gWm, Xpos, main_coef_pos[:, None])
        np.add.at(gV, Xpos, main_coef_pos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gWm, Xneg, main_coef_neg[:, None])
        np.add.at(gV, Xneg, main_coef_neg[:, None, None] * (Sneg[:, None, :] - Eneg))

        # aux task (pointwise BCE) contributions to shared V and W_aux
        np.add.at(gWa, Xpos, aux_coef_pos[:, None])
        np.add.at(gV, Xpos, aux_coef_pos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gWa, Xneg, aux_coef_neg[:, None])
        np.add.at(gV, Xneg, aux_coef_neg[:, None, None] * (Sneg[:, None, :] - Eneg))

        gb_aux = float(aux_coef_pos.sum() + aux_coef_neg.sum())
        # main bias cancels exactly in the BPR difference (same as single-task FM).

        gV = gV + self.l2 * self.V
        gWm = gWm + self.l2 * self.W_main
        gWa = gWa + self.l2 * self.W_aux
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                            (self.W_main, gWm, self.mWm, self.vWm),
                            (self.W_aux, gWa, self.mWa, self.vWa)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.mba = b1 * self.mba + (1 - b1) * gb_aux
        self.vba = b2 * self.vba + (1 - b2) * (gb_aux * gb_aux)
        self.b_aux -= self.lr * (self.mba / (1 - b1 ** self.t)) / (np.sqrt(self.vba / (1 - b2 ** self.t)) + eps)

        bpr_loss = float(np.mean(-np.log(p + 1e-9)))
        aux_bce = float(-np.mean(aux_pos * np.log(pa_pos + 1e-9) + (1 - aux_pos) * np.log(1 - pa_pos + 1e-9)))
        return bpr_loss + alpha * aux_bce

    def predict(self, X, bs=200_000):
        return np.concatenate([self.main_logits(X[i:i + bs]) for i in range(0, len(X), bs)])


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


def _build_neg_sampler(Xtr, ytr, rng_seed=0):
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

    return pos_row_idx, sample_negs


def run_fm_bpr(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, neg_k=1, seed=0, verbose=True):
    """BPR pairwise-ranking training loop (single-task), kept for reference / --model bpr."""
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pos_row_idx, sample_negs = _build_neg_sampler(Xtr, ytr)

    m = FM(dim, k=k, lr=lr, seed=seed)
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


def run_fm_bpr_mt(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, neg_k=1,
                   alpha_aux=0.1, seed=0, verbose=True):
    """Multi-task training: main head trained with BPR pairwise loss on long_view (same
    negative-sampling scheme as run_fm_bpr), auxiliary head trained with pointwise BCE
    on is_like for the same positive/negative rows, sharing the FM's embedding table V.
    Per ESMM/MMoE literature notes: the auxiliary signal regularizes the shared
    embeddings with a second, less-noisy gradient rather than just adding raw capacity."""
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    aux_tr = enc.get('train_aux')
    if aux_tr is None:
        aux_tr = np.zeros_like(ytr)
    pos_row_idx, sample_negs = _build_neg_sampler(Xtr, ytr)

    m = FM_MT(dim, k=k, lr=lr, seed=seed)
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
                losses.append(m.step_bpr_mt(Xtr[pos_batch], Xtr[neg_batch],
                                             aux_tr[pos_batch], aux_tr[neg_batch],
                                             alpha=alpha_aux))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W_main.copy(), np.float32(m.b_main),
                           m.W_aux.copy(), np.float32(m.b_aux))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W_main, m.b_main, m.W_aux, m.b_aux = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def run_model(splits, hparams, seed=0, verbose=True):
    """Generic entrypoint the harness (run_and_report.py) always calls, regardless
    of model family. Default: multi-task BPR + auxiliary is_like head (run_fm_bpr_mt)."""
    kwargs = dict(hparams)
    kwargs.pop('bpr', None)
    return run_fm_bpr_mt(splits, seed=seed, verbose=verbose, **kwargs)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='mt', choices=['pop', 'fm', 'bpr', 'mt', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--alpha_aux', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'bpr': lambda s: run_fm_bpr(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'mt': lambda s: run_fm_bpr_mt(s, k=a.k, lr=a.lr, epochs=a.epochs,
                                          alpha_aux=a.alpha_aux, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")