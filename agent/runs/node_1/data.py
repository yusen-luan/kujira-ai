"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket',
          'user_active_degree', 'follow_user_num_range', 'fans_user_num_range',
          'friend_user_num_range', 'register_days_range', 'is_live_streamer',
          'is_video_author']

_UNK = 'UNK'

def _clean_is_live_streamer(v):
    try:
        iv = int(v)
    except (ValueError, TypeError):
        return _UNK
    return str(iv) if iv >= 0 else _UNK

def load(data_dir):
    """读日志 + 视频侧特征 + 用户侧特征，返回按划分切好的 dict。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    uid2feat = {}
    with open(os.path.join(data_dir, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            uid2feat[r['user_id']] = {
                'user_active_degree': r.get('user_active_degree', _UNK) or _UNK,
                'follow_user_num_range': r.get('follow_user_num_range', _UNK) or _UNK,
                'fans_user_num_range': r.get('fans_user_num_range', _UNK) or _UNK,
                'friend_user_num_range': r.get('friend_user_num_range', _UNK) or _UNK,
                'register_days_range': r.get('register_days_range', _UNK) or _UNK,
                'is_live_streamer': _clean_is_live_streamer(r.get('is_live_streamer', '')),
                'is_video_author': r.get('is_video_author', _UNK) or _UNK,
            }
    default_ufeat = {
        'user_active_degree': _UNK, 'follow_user_num_range': _UNK,
        'fans_user_num_range': _UNK, 'friend_user_num_range': _UNK,
        'register_days_range': _UNK, 'is_live_streamer': _UNK, 'is_video_author': _UNK,
    }

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                uf = uid2feat.get(r['user_id'], default_ufeat)
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']),
                             uf['user_active_degree'], uf['follow_user_num_range'],
                             uf['fans_user_num_range'], uf['friend_user_num_range'],
                             uf['register_days_range'], uf['is_live_streamer'],
                             uf['is_video_author'],
                             1 if r[LABEL] != '0' else 0))

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
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))),
                x[6], x[7], x[8], x[9], x[10], x[11], x[12]]

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
            y[n] = x[13]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
