"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine，pairwise BPR loss（同用户内 正/负样本对比）
  --model mtfm  : Multi-task FM -- shared embeddings, main long_view BPR task + auxiliary
                  is_like pointwise-BCE task (per ESMM-style shared-embedding multi-task idea)
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


def run_fm(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
           neg_per_pos=2):
    neg_per_pos = max(1, int(round(neg_per_pos)))
    enc, dim = encode(splits)
    Xtr, ytr, users_tr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)

    pos_rows, pos_users, neg_rows_all, neg_offsets, neg_counts = _build_pair_index(
        ytr, users_tr, neg_per_pos, seed)
    n_pairs = len(pos_rows)

    best, best_state, bad = -1, None, 0
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


# ---------------- Multi-task FM: shared embeddings, main (long_view, BPR) + aux (is_like, BCE) ----------------
class MTFM:
    """Shared factorization-machine embedding table V used by two task heads:
    - main: long_view, scored/trained via pairwise BPR (same as plain FM), linear head W_main.
    - aux:  is_like, trained via pointwise BCE, linear head W_aux + its own bias b_aux.
    Only the main head (logits_main) is ever used for prediction/evaluation -- the aux task
    exists purely to add a denser gradient signal into the shared V, per ESMM-style
    shared-embedding multi-task learning (Ma et al. 2018)."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W_main = np.zeros(dim, dtype=np.float32)
        self.W_aux = np.zeros(dim, dtype=np.float32)
        self.b_aux = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mWm = np.zeros_like(self.W_main); self.vWm = np.zeros_like(self.W_main)
        self.mWa = np.zeros_like(self.W_aux); self.vWa = np.zeros_like(self.W_aux)
        self.mba = 0.0; self.vba = 0.0
        self.t = 0

    def _interact(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return inter, E, S

    def logits_main(self, X):
        inter, E, S = self._interact(X)
        return self.W_main[X].sum(1) + inter, E, S

    def logits_aux(self, X):
        inter, E, S = self._interact(X)
        return self.b_aux + self.W_aux[X].sum(1) + inter, E, S

    def step(self, Xpos, Xneg, Xaux, yaux, aux_weight):
        B = len(Xpos)
        zpos, Epos, Spos = self.logits_main(Xpos)
        zneg, Eneg, Sneg = self.logits_main(Xneg)
        diff = zpos - zneg
        p = sigmoid(diff)
        g = ((p - 1.0) / B).astype(np.float32)

        gV = np.zeros_like(self.V); gWm = np.zeros_like(self.W_main)
        np.add.at(gWm, Xpos, g[:, None])
        np.add.at(gWm, Xneg, -g[:, None])
        np.add.at(gV, Xpos, g[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, -g[:, None, None] * (Sneg[:, None, :] - Eneg))
        main_loss = float(-np.mean(np.log(p + 1e-9)))

        Ba = len(Xaux)
        zaux, Eaux, Saux = self.logits_aux(Xaux)
        paux = sigmoid(zaux)
        gaux = ((paux - yaux) / Ba).astype(np.float32)
        gWa = np.zeros_like(self.W_aux)
        gVa = np.zeros_like(self.V)
        np.add.at(gWa, Xaux, gaux[:, None])
        np.add.at(gVa, Xaux, gaux[:, None, None] * (Saux[:, None, :] - Eaux))
        gba = float(np.sum(gaux))
        aux_loss = float(-np.mean(yaux * np.log(paux + 1e-9) + (1 - yaux) * np.log(1 - paux + 1e-9)))

        gV_total = gV + aux_weight * gVa + self.l2 * self.V
        gWm_total = gWm + self.l2 * self.W_main
        gWa_total = gWa * aux_weight + self.l2 * self.W_aux
        gba_total = gba * aux_weight

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8

        def upd(P, G, M, Vv):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

        upd(self.V, gV_total, self.mV, self.vV)
        upd(self.W_main, gWm_total, self.mWm, self.vWm)
        upd(self.W_aux, gWa_total, self.mWa, self.vWa)

        self.mba = b1 * self.mba + (1 - b1) * gba_total
        self.vba = b2 * self.vba + (1 - b2) * (gba_total * gba_total)
        self.b_aux -= self.lr * (self.mba / (1 - b1 ** self.t)) / ((self.vba / (1 - b2 ** self.t)) ** 0.5 + eps)

        return main_loss, aux_loss

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits_main(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def run_mtfm(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
             neg_per_pos=2, aux_weight=0.2):
    neg_per_pos = max(1, int(round(neg_per_pos)))
    enc, dim = encode(splits)
    Xtr, ytr, users_tr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    aux_ytr = np.array([r[7] for r in splits['train']], dtype=np.float32)

    m = MTFM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)

    pos_rows, pos_users, neg_rows_all, neg_offsets, neg_counts = _build_pair_index(
        ytr, users_tr, neg_per_pos, seed)
    n_pairs = len(pos_rows)
    n_train = len(Xtr)

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        r = rng.integers(0, neg_counts[pos_users])
        neg_row_idx = neg_rows_all[neg_offsets[pos_users] + r]
        perm = rng.permutation(n_pairs)
        aux_idx_all = rng.integers(0, n_train, n_pairs)

        main_losses = []; aux_losses = []
        for i in range(0, n_pairs, bs):
            b_idx = perm[i:i + bs]
            Xpos = Xtr[pos_rows[b_idx]]
            Xneg = Xtr[neg_row_idx[b_idx]]
            a_idx = aux_idx_all[i:i + bs]
            Xaux = Xtr[a_idx]; yaux = aux_ytr[a_idx]
            ml, al = m.step(Xpos, Xneg, Xaux, yaux, aux_weight)
            main_losses.append(ml); aux_losses.append(al)

        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | main_loss {np.mean(main_losses):.4f} | aux_loss {np.mean(aux_losses):.4f} "
                  f"| valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W_main.copy())
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W_main = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def run_model(splits, hparams, seed=0, verbose=True):
    """Generic entrypoint the harness (run_and_report.py) always calls, regardless
    of model family. Default: multi-task FM (shared embeddings, long_view BPR main task +
    is_like BCE auxiliary task)."""
    return run_mtfm(splits, seed=seed, verbose=verbose, **hparams)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='mtfm', choices=['pop', 'fm', 'mtfm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--neg_per_pos', type=int, default=2)
    ap.add_argument('--aux_weight', type=float, default=0.2)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, l2=a.l2, epochs=a.epochs, seed=a.seed,
                                   neg_per_pos=a.neg_per_pos),
           'mtfm': lambda s: run_mtfm(s, k=a.k, lr=a.lr, l2=a.l2, epochs=a.epochs, seed=a.seed,
                                       neg_per_pos=a.neg_per_pos, aux_weight=a.aux_weight)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")