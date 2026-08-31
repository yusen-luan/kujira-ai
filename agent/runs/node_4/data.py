"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 6 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'vid_pop_bucket']

VID_QUALITY_PRIOR = 20.0  # smoothing strength, matches run_pop's default prior in baseline.py


def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]

    # ---- video historical engagement-quality feature ----
    # Computed ONLY from train-split history (never from valid/test, which are
    # strictly later in time). For train rows themselves we use leave-one-out
    # (exclude the row's own label) so a row can never see its own outcome.
    pos_v, cnt_v = collections.Counter(), collections.Counter()
    for x in out['train']:
        cnt_v[x[2]] += 1
        pos_v[x[2]] += x[6]
    total_pos = sum(pos_v.values())
    total_cnt = sum(cnt_v.values())
    gmean = (total_pos / total_cnt) if total_cnt > 0 else 0.5
    prior = VID_QUALITY_PRIOR

    def rate_loo(v, y):
        c, p = cnt_v[v], pos_v[v]
        return (p - y + prior * gmean) / (c - 1 + prior)

    def rate_full(v):
        c, p = cnt_v.get(v, 0), pos_v.get(v, 0)
        return (p + prior * gmean) / (c + prior) if c > 0 else gmean

    for name in out:
        new_rows = []
        for x in out[name]:
            date, u, v, a, tab, dur, y = x
            q = rate_loo(v, y) if name == 'train' else rate_full(v)
            new_rows.append((date, u, v, a, tab, dur, y, q))
        out[name] = new_rows

    return out


def _bucket_edges(vals, n=10):
    return np.quantile(np.asarray(vals), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])
    q_edges = _bucket_edges([x[7] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4],
                str(int(np.searchsorted(dur_edges, x[5]))),
                str(int(np.searchsorted(q_edges, x[7])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))