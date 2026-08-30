"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 6 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'hist_bucket']

HIST_PRIOR = 10.0  # additive-smoothing strength for the historical-rate feature


def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。
    额外计算一个"用户历史 long_view 率"特征：对每一行，只使用该用户在此次
    曝光**之前**（按 time_ms 严格排序）发生过的曝光来估计其历史正例率，
    做加性平滑后分桶。这是纯历史聚合统计量，不会看到当前行自己的标签，
    因此不构成 leakage。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    raw_rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                raw_rows.append((int(r['date']), r['user_id'], r['video_id'],
                                  vid2author.get(r['video_id'], 'UNK'), r['tab'],
                                  float(r['duration_ms']), int(r['time_ms']),
                                  1 if r[LABEL] != '0' else 0))

    # global mean long_view rate computed from train-window rows only, used
    # purely as a smoothing prior constant (not per-row label leakage).
    lo_tr, hi_tr = SPLITS['train']
    tr_labels = [x[7] for x in raw_rows if lo_tr <= x[0] <= hi_tr]
    global_mean = float(np.mean(tr_labels)) if tr_labels else 0.3

    # chronological order by absolute timestamp (stable sort keeps tie order stable)
    times = np.array([x[6] for x in raw_rows], dtype=np.int64)
    order = np.argsort(times, kind='stable')

    stats = collections.defaultdict(lambda: [0.0, 0.0])  # user -> [pos, imp]
    hist_rate = [0.0] * len(raw_rows)
    for i in order:
        i = int(i)
        uid = raw_rows[i][1]
        lbl = raw_rows[i][7]
        pos, imp = stats[uid]
        hist_rate[i] = (pos + HIST_PRIOR * global_mean) / (imp + HIST_PRIOR)
        s = stats[uid]
        s[0] += lbl
        s[1] += 1.0

    rows = [(raw_rows[i][0], raw_rows[i][1], raw_rows[i][2], raw_rows[i][3],
              raw_rows[i][4], raw_rows[i][5], hist_rate[i], raw_rows[i][7])
             for i in range(len(raw_rows))]

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(vals, n=10):
    return np.quantile(np.asarray(vals), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])
    hist_edges = _bucket_edges([x[6] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4],
                str(int(np.searchsorted(dur_edges, x[5]))),
                str(int(np.searchsorted(hist_edges, x[6])))]

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
            y[n] = x[7]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))