"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

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
    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

# Fields subject to rare-id collapsing: user_id, video_id, author_id (the three
# high-cardinality id-like fields). tab/dur_bucket are already small, designed
# categoricals and are left untouched by the threshold.
_THRESH_FIELDS = {0, 1, 2}

def encode(splits, min_count=1):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    For fields in _THRESH_FIELDS, any train-time value occurring fewer than
    `min_count` times is instead collapsed into a shared per-field RARE bucket
    (distinct from the UNK bucket used for values never seen in train at all).
    This caps the embedding-table capacity spent memorizing very-low-frequency
    ids, which are the noisiest to learn well. min_count=1 (default) reproduces
    the original one-id-per-value behavior exactly (no collapsing).
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    counts = [collections.Counter() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            counts[i][v] += 1

    vocabs = [dict() for _ in FIELDS]
    rare_sets = [set() for _ in FIELDS]
    for i in range(len(FIELDS)):
        if i in _THRESH_FIELDS and min_count > 1:
            for v, c in counts[i].items():
                if c >= min_count:
                    vocabs[i][v] = len(vocabs[i])
                else:
                    rare_sets[i].add(v)
            if rare_sets[i]:
                vocabs[i]['__RARE__'] = len(vocabs[i])
        else:
            for v in counts[i]:
                vocabs[i][v] = len(vocabs[i])

    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽 (true cold-start)
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    def lookup(i, v):
        vi = vocabs[i]
        if v in vi:
            return vi[v]
        if v in rare_sets[i]:
            return vi['__RARE__']
        return unk[i]

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = lookup(i, v) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))