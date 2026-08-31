import numpy as np


def run_probe(arrays, masks, user_feat, video_feat, label):
    train_mask = masks['train']
    valid_mask = masks['valid']

    users = arrays['user_id']
    videos = arrays['video_id']
    y = arrays[label]

    # --- fixed scoring function: smoothed train-set item popularity (same form as run_pop) ---
    train_videos = videos[train_mask]
    train_y = y[train_mask].astype(np.float64)
    uniq_v, inv = np.unique(train_videos, return_inverse=True)
    imp = np.bincount(inv).astype(np.float64)
    pos = np.bincount(inv, weights=train_y)
    prior = 20.0
    gmean = pos.sum() / imp.sum()
    vid_to_idx = {int(v): i for i, v in enumerate(uniq_v)}

    def pop_score(vid_arr):
        idx = np.fromiter((vid_to_idx.get(int(v), -1) for v in vid_arr), dtype=np.int64, count=len(vid_arr))
        s = np.full(len(vid_arr), gmean, dtype=np.float64)
        known = idx >= 0
        s[known] = (pos[idx[known]] + prior * gmean) / (imp[idx[known]] + prior)
        return s

    vu = users[valid_mask]
    vv = videos[valid_mask]
    vy = y[valid_mask].astype(np.float64)
    vs = pop_score(vv)

    # --- group rows by user ---
    order = np.argsort(vu, kind='stable')
    vu_s, vy_s, vs_s = vu[order], vy[order], vs[order]
    uniq_u, start_idx, counts = np.unique(vu_s, return_index=True, return_counts=True)
    n_users = len(uniq_u)

    user_auc = np.full(n_users, np.nan, dtype=np.float64)
    user_weight = np.zeros(n_users, dtype=np.float64)
    user_ndcg = np.zeros(n_users, dtype=np.float64)

    for i in range(n_users):
        st = start_idx[i]
        c = int(counts[i])
        ys = vy_s[st:st + c]
        ss = vs_s[st:st + c]
        npos = ys.sum()
        nneg = c - npos
        user_weight[i] = c

        if npos > 0 and nneg > 0:
            ordr = np.argsort(ss, kind='stable')
            ranks = np.empty(c, dtype=np.float64)
            ranks[ordr] = np.arange(1, c + 1, dtype=np.float64)
            sum_rank_pos = ranks[ys > 0].sum()
            user_auc[i] = (sum_rank_pos - npos * (npos + 1) / 2.0) / (npos * nneg)

        k = min(5, c)
        top_idx = np.argsort(-ss, kind='stable')[:k]
        rel = ys[top_idx]
        discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
        dcg = float((rel * discounts).sum())
        ideal_rel = np.sort(ys)[::-1][:k]
        idcg = float((ideal_rel * discounts).sum())
        user_ndcg[i] = dcg / idcg if idcg > 0 else 0.0

    elig_mask = ~np.isnan(user_auc)
    elig_idx = np.nonzero(elig_mask)[0]
    elig_auc = user_auc[elig_idx]
    elig_w = user_weight[elig_idx]
    n_elig = len(elig_idx)

    gauc_point = float(np.average(elig_auc, weights=elig_w))
    ndcg_point = float(user_ndcg.mean())
    primary_point = 0.5 * (gauc_point + ndcg_point)

    rng = np.random.default_rng(0)
    n_boot = 500
    boot_gauc = np.empty(n_boot)
    boot_ndcg = np.empty(n_boot)
    for b in range(n_boot):
        samp1 = rng.integers(0, n_elig, n_elig)
        boot_gauc[b] = np.average(elig_auc[samp1], weights=elig_w[samp1])
        samp2 = rng.integers(0, n_users, n_users)
        boot_ndcg[b] = user_ndcg[samp2].mean()
    boot_primary = 0.5 * (boot_gauc + boot_ndcg)

    return {
        'n_valid_users': int(n_users),
        'n_eligible_gauc_users': int(n_elig),
        'frac_eligible_gauc_users': float(n_elig / n_users),
        'gauc_point_popularity_baseline': gauc_point,
        'ndcg_point_popularity_baseline': ndcg_point,
        'primary_point_popularity_baseline': primary_point,
        'gauc_bootstrap_std': float(boot_gauc.std()),
        'ndcg_bootstrap_std': float(boot_ndcg.std()),
        'primary_bootstrap_std': float(boot_primary.std()),
        'primary_bootstrap_p5': float(np.percentile(boot_primary, 5)),
        'primary_bootstrap_p95': float(np.percentile(boot_primary, 95)),
        'primary_bootstrap_range_p5_p95': float(np.percentile(boot_primary, 95) - np.percentile(boot_primary, 5)),
    }