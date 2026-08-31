"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model dfm   : DeepFM-lite (FM + small MLP head, pure numpy)
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

# ---------------- Factorization Machine (kept for CLI/back-compat) ----------------
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

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

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

# ---------------- DeepFM-lite: FM (linear + 2-way) + small MLP head, plain numpy ----------------
class DeepFM:
    def __init__(self, dim, k=16, h=32, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.dim, self.k, self.h = dim, k, h
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.rng = rng
        self.F = None  # number of fields, discovered lazily from first batch
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0
        self._adam_mlp = {}

    def _init_mlp(self, F):
        self.F = F
        in_dim = F * self.k
        self.W1 = self.rng.normal(0, 1.0 / np.sqrt(in_dim), (in_dim, self.h)).astype(np.float32)
        self.b1 = np.zeros(self.h, dtype=np.float32)
        self.W2 = self.rng.normal(0, 1.0 / np.sqrt(self.h), (self.h, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        for name in ('W1', 'b1', 'W2', 'b2'):
            p = getattr(self, name)
            self._adam_mlp[name] = [np.zeros_like(p), np.zeros_like(p)]

    def forward(self, X):
        if self.F is None:
            self._init_mlp(X.shape[1])
        E = self.V[X]                                    # (B,F,k)
        S = E.sum(1)                                      # (B,k)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        linear = self.b + self.W[X].sum(1)
        B = X.shape[0]
        embed_flat = E.reshape(B, self.F * self.k)
        z1 = embed_flat @ self.W1 + self.b1               # (B,h)
        h1 = np.maximum(z1, 0)
        mlp_out = (h1 @ self.W2 + self.b2).reshape(-1)     # (B,)
        logit = linear + fm_inter + mlp_out
        return logit, E, S, embed_flat, h1, z1

    def _adam_update(self, P, G, M, Vv):
        b1c, b2c, eps = 0.9, 0.999, 1e-8
        M *= b1c; M += (1 - b1c) * G
        Vv *= b2c; Vv += (1 - b2c) * (G * G)
        P -= self.lr * (M / (1 - b1c ** self.t)) / (np.sqrt(Vv / (1 - b2c ** self.t)) + eps)

    def step(self, X, y):
        B = len(y)
        z, E, S, embed_flat, h1, z1 = self.forward(X)
        p = sigmoid(z)
        g = ((p - y) / B).astype(np.float32)               # (B,)

        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))

        gW2 = h1.T @ g[:, None]                             # (h,1)
        gb2 = np.array([g.sum()], dtype=np.float32)
        dh1 = g[:, None] @ self.W2.T                        # (B,h)
        dz1 = dh1 * (z1 > 0)
        gW1 = embed_flat.T @ dz1                            # (F*k,h)
        gb1 = dz1.sum(0)
        dembed = (dz1 @ self.W1.T).reshape(B, self.F, self.k)
        np.add.at(gV, X, dembed)

        gV += self.l2 * self.V; gW += self.l2 * self.W
        gW1 = gW1 + self.l2 * self.W1
        gW2 = gW2 + self.l2 * self.W2

        self.t += 1
        self._adam_update(self.V, gV, self.mV, self.vV)
        self._adam_update(self.W, gW, self.mW, self.vW)
        self.b -= self.lr * g.sum()
        for name, G in (('W1', gW1), ('b1', gb1), ('W2', gW2), ('b2', gb2)):
            P = getattr(self, name)
            M, Vv = self._adam_mlp[name]
            self._adam_update(P, G, M, Vv)

        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def predict(self, X, bs=200_000):
        outs = []
        for i in range(0, len(X), bs):
            z, *_ = self.forward(X[i:i + bs])
            outs.append(z)
        return np.concatenate(outs)


def run_dfm(splits, k=16, mlp_hidden=32, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = DeepFM(dim, k=k, h=mlp_hidden, lr=lr, l2=l2, seed=seed)
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
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b),
                           m.W1.copy(), m.b1.copy(), m.W2.copy(), m.b2.copy())
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b, m.W1, m.b1, m.W2, m.b2 = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}

def run_model(splits, hparams, seed=0, verbose=True):
    """Generic entrypoint the harness (run_and_report.py) always calls, regardless
    of model family. Default: DeepFM-lite (FM + small MLP head), reading hparams
    generically via .get with sensible defaults so unrelated/missing keys are safe."""
    kwargs = dict(
        k=hparams.get('k', 16),
        mlp_hidden=hparams.get('mlp_hidden', 32),
        lr=hparams.get('lr', 0.001),
        l2=hparams.get('l2', 1e-6),
        epochs=hparams.get('epochs', 40),
        bs=hparams.get('bs', 8192),
        patience=hparams.get('patience', 4),
    )
    return run_dfm(splits, seed=seed, verbose=verbose, **kwargs)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='dfm', choices=['pop', 'fm', 'dfm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--mlp_hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'dfm': lambda s: run_dfm(s, k=a.k, mlp_hidden=a.mlp_hidden, lr=a.lr, epochs=a.epochs, seed=a.seed),
           }[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")