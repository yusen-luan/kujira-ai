import numpy as np

def run_probe(arrays, masks, user_feat, video_feat, label):
    out = {}
    for split in ('train', 'valid', 'test'):
        m = masks[split]
        u = arrays['user_id'][m]
        y = arrays[label][m].astype(np.float64)

        uniq, inv = np.unique(u, return_inverse=True)
        n_users = len(uniq)
        counts = np.bincount(inv, minlength=n_users)
        pos_counts = np.bincount(inv, weights=y, minlength=n_users)

        mixed = (pos_counts > 0) & (pos_counts < counts)
        all_pos = pos_counts == counts
        all_neg = pos_counts == 0
        rows_mixed = int(counts[mixed].sum())

        out[f'{split}_n_users'] = int(n_users)
        out[f'{split}_n_rows'] = int(len(u))
        out[f'{split}_mean_rows_per_user'] = float(counts.mean())
        out[f'{split}_median_rows_per_user'] = float(np.median(counts))
        out[f'{split}_frac_users_mixed_label'] = float(mixed.mean())
        out[f'{split}_frac_rows_from_mixed_users'] = float(rows_mixed / len(u)) if len(u) else 0.0
        out[f'{split}_frac_users_all_positive'] = float(all_pos.mean())
        out[f'{split}_frac_users_all_negative'] = float(all_neg.mean())
        out[f'{split}_frac_users_lt5_rows'] = float((counts < 5).mean())
        out[f'{split}_pos_rate'] = float(y.mean())

    return out