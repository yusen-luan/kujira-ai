"""KuaiRand-Pure baselines.
  --model pop   : item popularity (official baseline, no training)
  --model fm    : Factorization Machine, pairwise BPR loss (in-user pos/neg contrast)
  --model random: random scoring (lower bound sanity check)
run_model() (the harness entrypoint) trains TWO structurally different BPR-trained
scorers -- a plain bilinear FM (numpy, exact mechanism of the current accepted best)
and a DeepFM (bilinear FM term + MLP over the same shared embeddings, torch) -- and
ensembles their standardized scores. Needs numpy + torch only.
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity (official baseline) ----------------
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

# ---------------- Factorization Machine (pairwise BPR training, numpy) ----------------
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


def _train_fm(Xtr, ytr, users_tr, Xva, yva, uva, dim, k=16, lr=0.001, l2=1e-6,
              epochs=40, bs=8192, patience=4, neg_per_pos=1, seed=0, verbose=True, tag='fm'):
    neg_per_pos = max(1, int(round(neg_per_pos)))
    m = FM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    pos_rows, pos_users, neg_rows_all, neg_offsets, neg_counts = _build_pair_index(
        ytr, users_tr, neg_per_pos, seed)
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

        va = evaluate(uva, yva, m.predict(Xva))
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


def run_fm(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
           neg_per_pos=1):
    enc, dim = encode(splits)
    Xtr, ytr, users_tr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = _train_fm(Xtr, ytr, users_tr, Xva, yva, uva, dim, k=k, lr=lr, l2=l2, epochs=epochs,
                  bs=bs, patience=patience, neg_per_pos=neg_per_pos, seed=seed, verbose=verbose)
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


# ---------------- DeepFM (bilinear FM term + MLP over shared embeddings, torch) ----------------
if torch is not None:
    class TorchDeepFM(nn.Module):
        """Global-vocab DeepFM: shares the same flat (dim,k) embedding table convention
        as the numpy FM above (X already carries per-field global offsets from encode()),
        adds a small MLP over the concatenated field embeddings for non-linear interactions."""
        def __init__(self, dim, num_fields, k=16, mlp_dims=(16,), dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(dim, k)
            self.linear = nn.Embedding(dim, 1)
            self.bias = nn.Parameter(torch.zeros(1))
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)
            layers = []
            in_dim = num_fields * k
            for h in mlp_dims:
                layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = h
            layers += [nn.Linear(in_dim, 1)]
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            e = self.embedding(x)                      # (B,F,k)
            s = e.sum(1)
            inter = 0.5 * ((s ** 2).sum(1) - (e ** 2).sum((1, 2)))
            lin = self.linear(x).sum(1).squeeze(-1)
            mlp_out = self.mlp(e.reshape(e.size(0), -1)).squeeze(-1)
            return self.bias + lin + inter + mlp_out


def _train_deepfm_torch(Xtr, ytr, users_tr, Xva, yva, uva, dim, num_fields,
                         k=16, mlp_dims=(16,), dropout=0.3, lr=0.001, wd=1e-6,
                         epochs=10, bs=8192, patience=3, neg_per_pos=1, seed=0, verbose=True):
    if torch is None:
        raise RuntimeError("torch not available")
    torch.manual_seed(seed)
    neg_per_pos = max(1, int(round(neg_per_pos)))
    model = TorchDeepFM(dim, num_fields, k=k, mlp_dims=mlp_dims, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    pos_rows, pos_users, neg_rows_all, neg_offsets, neg_counts = _build_pair_index(
        ytr, users_tr, neg_per_pos, seed)
    n_pairs = len(pos_rows)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    Xva_t = torch.from_numpy(Xva.astype(np.int64))

    best, bad = -1, 0
    best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
    for ep in range(1, epochs + 1):
        t0 = time.time()
        r = rng.integers(0, neg_counts[pos_users])
        neg_row_idx = neg_rows_all[neg_offsets[pos_users] + r]
        perm = rng.permutation(n_pairs)

        model.train()
        losses = []
        for i in range(0, n_pairs, bs):
            b_idx = perm[i:i + bs]
            xp = Xtr_t[pos_rows[b_idx]]
            xn = Xtr_t[neg_row_idx[b_idx]]
            zp = model(xp); zn = model(xn)
            loss = -F.logsigmoid(zp - zn).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            va_pred = model(Xva_t).numpy()
        va = evaluate(uva, yva, va_pred)
        if verbose:
            print(f"  [dfm] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  [dfm] early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _torch_predict(model, X, bs=200_000):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64))
            out.append(model(xb).numpy())
    return np.concatenate(out)


def _standardize(scores, mean, std):
    return (scores - mean) / (std + 1e-8)


# ---------------- Ensemble: bilinear BPR-FM + BPR-DeepFM ----------------
def run_ensemble(splits, hparams, seed=0, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, users_tr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    num_fields = Xtr.shape[1]

    # branch A: plain bilinear BPR-FM (mechanism of the current accepted best)
    fm = _train_fm(Xtr, ytr, users_tr, Xva, yva, uva, dim,
                    k=hparams.get('k', 16), lr=hparams.get('lr', 0.001),
                    l2=hparams.get('l2', 1e-6), epochs=hparams.get('epochs', 40),
                    bs=hparams.get('bs', 8192), patience=hparams.get('patience', 4),
                    neg_per_pos=hparams.get('neg_per_pos', 1), seed=seed, verbose=verbose, tag='fm')
    fm_va = fm.predict(Xva); fm_te = fm.predict(Xte)

    if torch is None:
        # torch unavailable -- fall back to the FM branch alone
        return {'valid': evaluate(uva, yva, fm_va), 'test': evaluate(ute, yte, fm_te)}

    # branch B: DeepFM (bilinear FM term + MLP), structurally different scorer
    dfm = _train_deepfm_torch(Xtr, ytr, users_tr, Xva, yva, uva, dim, num_fields,
                               k=hparams.get('dfm_k', 16),
                               mlp_dims=tuple(hparams.get('dfm_mlp_dims', (16,))),
                               dropout=hparams.get('dfm_dropout', 0.3),
                               lr=hparams.get('dfm_lr', 0.001),
                               wd=hparams.get('dfm_wd', 1e-6),
                               epochs=hparams.get('dfm_epochs', 10),
                               bs=hparams.get('dfm_bs', 8192),
                               patience=hparams.get('dfm_patience', 3),
                               neg_per_pos=hparams.get('dfm_neg_per_pos', 1),
                               seed=seed, verbose=verbose)
    dfm_va = _torch_predict(dfm, Xva); dfm_te = _torch_predict(dfm, Xte)

    # standardize using valid-split stats, apply same affine transform to test
    fm_mean, fm_std = fm_va.mean(), fm_va.std()
    dfm_mean, dfm_std = dfm_va.mean(), dfm_va.std()

    ens_w = float(hparams.get('ens_weight', 0.5))
    ens_va = ens_w * _standardize(fm_va, fm_mean, fm_std) + (1 - ens_w) * _standardize(dfm_va, dfm_mean, dfm_std)
    ens_te = ens_w * _standardize(fm_te, fm_mean, fm_std) + (1 - ens_w) * _standardize(dfm_te, dfm_mean, dfm_std)

    return {'valid': evaluate(uva, yva, ens_va), 'test': evaluate(ute, yte, ens_te)}


def run_model(splits, hparams, seed=0, verbose=True):
    """Generic entrypoint the harness (run_and_report.py) always calls, regardless
    of model family. Trains both a bilinear BPR-FM and a BPR-DeepFM and ensembles
    their standardized scores (see run_ensemble)."""
    return run_ensemble(splits, hparams, seed=seed, verbose=verbose)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure data directory')
    ap.add_argument('--model', default='ensemble', choices=['pop', 'fm', 'random', 'ensemble'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--neg_per_pos', type=int, default=1)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': lambda s: run_pop(s), 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                                   neg_per_pos=a.neg_per_pos),
           'ensemble': lambda s: run_ensemble(s, {}, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")