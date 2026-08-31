"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : DeepFM (FM + MLP interaction), pairwise BPR loss（同用户内 正/负样本对比）
  --model random: 随机打分（下界，用来自检评测代码没坏）
numpy + torch/torchfm allowed here. 用法见 README.md
"""
import argparse, collections, time
import numpy as np
import torch
import torch.nn as nn
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

# ---------------- DeepFM (FM term + MLP interaction), pairwise BPR training ----------------
class DeepFM(nn.Module):
    """Self-contained DeepFM: global-offset embedding table (matches data.encode's
    single combined `dim`), linear term, bilinear FM interaction, plus an MLP tower
    over the flattened field embeddings for higher-order interactions."""
    def __init__(self, dim, num_fields, k=16, mlp_dims=(32, 16), dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(dim, k)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.linear = nn.Embedding(dim, 1)
        nn.init.zeros_(self.linear.weight)
        self.bias = nn.Parameter(torch.zeros(1))

        layers = []
        in_dim = num_fields * k
        for h in mlp_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """x: LongTensor (B, num_fields), entries already globally offset per field."""
        e = self.embedding(x)                       # (B, F, k)
        s = e.sum(1)                                 # (B, k)
        inter = 0.5 * ((s ** 2).sum(1) - (e ** 2).sum((1, 2)))   # (B,)
        lin = self.linear(x).sum(1).squeeze(-1)       # (B,)
        mlp_out = self.mlp(e.reshape(e.size(0), -1)).squeeze(-1)  # (B,)
        return self.bias.squeeze(0) + lin + inter + mlp_out


def bpr_loss(zpos, zneg):
    diff = zpos - zneg
    return -torch.log(torch.sigmoid(diff) + 1e-9).mean()


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


def run_fm(splits, k=16, lr=0.001, l2=1e-6, epochs=15, bs=8192, patience=3, seed=0, verbose=True,
           neg_per_pos=2, dropout=0.2, mlp_dims=(32, 16)):
    neg_per_pos = max(1, int(round(neg_per_pos)))
    enc, dim = encode(splits)
    Xtr, ytr, users_tr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    num_fields = Xtr.shape[1]

    torch.manual_seed(seed)
    model = DeepFM(dim, num_fields, k=k, mlp_dims=list(mlp_dims), dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    rng = np.random.default_rng(seed)
    pos_rows, pos_users, neg_rows_all, neg_offsets, neg_counts = _build_pair_index(
        ytr, users_tr, neg_per_pos, seed)
    n_pairs = len(pos_rows)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    Xva_t = torch.from_numpy(Xva.astype(np.int64))
    Xte_t = torch.from_numpy(Xte.astype(np.int64))
    pos_rows_t = torch.from_numpy(pos_rows.astype(np.int64))

    def predict(X_t, bs_pred=200_000):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X_t), bs_pred):
                outs.append(model(X_t[i:i + bs_pred]).numpy())
        return np.concatenate(outs)

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        r = rng.integers(0, neg_counts[pos_users])
        neg_row_idx = neg_rows_all[neg_offsets[pos_users] + r]
        neg_row_idx_t = torch.from_numpy(neg_row_idx.astype(np.int64))
        perm = torch.from_numpy(rng.permutation(n_pairs).astype(np.int64))

        model.train()
        losses = []
        for i in range(0, n_pairs, bs):
            b_idx = perm[i:i + bs]
            xp = Xtr_t[pos_rows_t[b_idx]]
            xn = Xtr_t[neg_row_idx_t[b_idx]]
            zpos = model(xp)
            zneg = model(xn)
            loss = bpr_loss(zpos, zneg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, predict(Xva_t))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return {'valid': evaluate(uva, yva, predict(Xva_t)),
            'test':  evaluate(ute, yte, predict(Xte_t))}

def run_model(splits, hparams, seed=0, verbose=True):
    """Generic entrypoint the harness (run_and_report.py) always calls, regardless
    of model family. Default: unpack hparams as run_fm's keyword args. A candidate
    that swaps in a different model should replace this function's body (keep the
    name/signature) rather than renaming run_fm itself."""
    return run_fm(splits, seed=seed, verbose=verbose, **hparams)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--neg_per_pos', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.2)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, l2=a.l2, epochs=a.epochs, seed=a.seed,
                                   neg_per_pos=a.neg_per_pos, dropout=a.dropout)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")