"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 6 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'pos_bucket']

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。
    每行额外附带一个因果的、逐日会话内位置计数（pos_bucket）：
    该用户当天在该次曝光之前（含本次）已经看过多少条视频，封顶在 20。
    这只用到本次曝光及更早的时间戳信息，服务时点完全可知，不构成标签泄漏。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    dates, users, vids, authors, tabs, durs, labels, hourmins, times = (
        [], [], [], [], [], [], [], [], [])
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                dates.append(int(r['date']))
                users.append(r['user_id'])
                vids.append(r['video_id'])
                authors.append(vid2author.get(r['video_id'], 'UNK'))
                tabs.append(r['tab'])
                durs.append(float(r['duration_ms']))
                labels.append(1 if r[LABEL] != '0' else 0)
                hourmins.append(int(r['hourmin']))
                times.append(int(r['time_ms']))

    n = len(dates)
    dates_a = np.asarray(dates, dtype=np.int64)
    users_a = np.asarray(users)
    hourmins_a = np.asarray(hourmins, dtype=np.int64)
    times_a = np.asarray(times, dtype=np.int64)

    _, user_codes = np.unique(users_a, return_inverse=True)
    # primary key = user, then date, then time-of-day (hourmin, then time_ms)
    order = np.lexsort((times_a, hourmins_a, dates_a, user_codes))
    uc_sorted = user_codes[order]
    d_sorted = dates_a[order]
    grp_key = uc_sorted.astype(np.int64) * 100000 + (d_sorted - d_sorted.min())
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = grp_key[1:] != grp_key[:-1]
    grp_id = np.cumsum(change) - 1
    first_occurrence = np.nonzero(change)[0]
    start_idx_sorted = first_occurrence[grp_id]
    pos_sorted = np.arange(n) - start_idx_sorted

    pos = np.zeros(n, dtype=np.int64)
    pos[order] = pos_sorted
    pos_bucket = np.minimum(pos, 20)

    rows = []
    for i in range(n):
        rows.append((dates[i], users[i], vids[i], authors[i], tabs[i], durs[i],
                     labels[i], int(pos_bucket[i])))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))), str(x[7])]

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