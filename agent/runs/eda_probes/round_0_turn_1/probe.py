def run_probe(arrays, masks, user_feat, video_feat, label):
    import numpy as np

    y = arrays[label]
    uid = arrays['user_id']
    time_ms = arrays['time_ms']
    train_mask = masks['train']
    valid_mask = masks['valid']

    # ---- (a) user historical rate -> valid bucket effect ----
    train_uid = uid[train_mask]
    train_y = y[train_mask].astype(np.float64)

    order = np.argsort(train_uid, kind='stable')
    sorted_uid = train_uid[order]
    sorted_y = train_y[order]
    unique_uid, start_idx, counts = np.unique(sorted_uid, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_y, start_idx)
    rates = sums / counts

    keep = counts >= 10
    u_ids = unique_uid[keep]
    u_rates = rates[keep]

    quantiles = np.quantile(u_rates, [0.2, 0.4, 0.6, 0.8])
    bucket_idx = np.searchsorted(quantiles, u_rates, side='right')  # 0..4
    user_to_bucket = dict(zip(u_ids.tolist(), bucket_idx.tolist()))

    valid_uid = uid[valid_mask]
    valid_y = y[valid_mask].astype(np.float64)
    bucket_for_valid = np.fromiter(
        (user_to_bucket.get(u, -1) for u in valid_uid), dtype=np.int64, count=len(valid_uid)
    )

    results_bucket = {}
    for b in range(5):
        sel = bucket_for_valid == b
        n = int(sel.sum())
        if n > 0:
            results_bucket[str(b)] = {'n': n, 'rate': float(valid_y[sel].mean())}

    covered = int((bucket_for_valid >= 0).sum())

    # ---- (b) within-user session position effect (train) ----
    train_time = time_ms[train_mask]
    order2 = np.lexsort((train_time, train_uid))
    sorted_uid2 = train_uid[order2]
    sorted_y2 = train_y[order2]

    _, start_idx2, counts2 = np.unique(sorted_uid2, return_index=True, return_counts=True)
    pos = np.empty(len(sorted_uid2), dtype=np.int64)
    for s, c in zip(start_idx2, counts2):
        pos[s:s + c] = np.arange(c)

    bin_edges = [1, 3, 6, 11, 21]
    bin_labels = ['0', '1-2', '3-5', '6-10', '11-20', '21+']
    bin_idx = np.digitize(pos, bin_edges, right=False)

    pos_results = {}
    for i, lbl in enumerate(bin_labels):
        sel = bin_idx == i
        n = int(sel.sum())
        if n > 0:
            pos_results[lbl] = {'n': n, 'rate': float(sorted_y2[sel].mean())}

    return {
        'user_hist_rate_bucket_valid': results_bucket,
        'valid_rows_covered_by_user_bucket': covered,
        'valid_rows_total': int(valid_mask.sum()),
        'session_position_rate_train': pos_results,
    }