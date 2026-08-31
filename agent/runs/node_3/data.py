"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket',
          'vid_pop_bucket', 'author_pop_bucket']

_POP_ALPHA = 20.0  # Bayesian-smoothing prior strength for popularity rate estimates


def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。
    额外计算“训练期内”视频/作者的平滑正例率（贝叶斯收缩），作为两个新的历史聚合特征，
    严格只用训练期数据估计，避免使用同一行的标签（不构成 leakage）。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    raw_rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                date = int(r['date'])
                vid = r['video_id']
                author = vid2author.get(vid, 'UNK')
                label = 1 if r[LABEL] != '0' else 0
                raw_rows.append((date, r['user_id'], vid, author, r['tab'],
                                  float(r['duration_ms']), label))

    lo_tr, hi_tr = SPLITS['train']
    vid_pos, vid_cnt = collections.Counter(), collections.Counter()
    auth_pos, auth_cnt = collections.Counter(), collections.Counter()
    tot_pos = tot_cnt = 0
    for date, uid, vid, author, tab, dur, label in raw_rows:
        if lo_tr <= date <= hi_tr:
            vid_cnt[vid] += 1; vid_pos[vid] += label
            auth_cnt[author] += 1; auth_pos[author] += label
            tot_cnt += 1; tot_pos += label
    gmean = (tot_pos / tot_cnt) if tot_cnt else 0.0

    def vid_rate(v):
        return (vid_pos[v] + _POP_ALPHA * gmean) / (vid_cnt[v] + _POP_ALPHA)

    def auth_rate(a):
        return (auth_pos[a] + _POP_ALPHA * gmean) / (auth_cnt[a] + _POP_ALPHA)

    rows = []
    for date, uid, vid, author, tab, dur, label in raw_rows:
        rows.append((date, uid, vid, author, tab, dur,
                      vid_rate(vid), auth_rate(author), label))

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
    vid_edges = _bucket_edges([x[6] for x in tr])
    auth_edges = _bucket_edges([x[7] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4],
                str(int(np.searchsorted(dur_edges, x[5]))),
                str(int(np.searchsorted(vid_edges, x[6]))),
                str(int(np.searchsorted(auth_edges, x[7])))]

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
            y[n] = x[8]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))