def run_probe(arrays, masks, user_feat, video_feat, label):
    import numpy as np
    tr = masks['train']
    dur = arrays['duration_ms'][tr].astype(np.float64)
    y = arrays[label][tr].astype(np.float64)

    n = len(dur)
    order = np.argsort(dur)
    dur_sorted = dur[order]
    y_sorted = y[order]

    n_bins = 10
    edges = np.quantile(dur, np.linspace(0, 1, n_bins + 1))
    # avoid duplicate edges issues
    bin_idx = np.searchsorted(edges, dur, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    result = {}
    rates = []
    counts = []
    dur_means = []
    for b in range(n_bins):
        mask_b = bin_idx == b
        cnt = int(mask_b.sum())
        if cnt == 0:
            continue
        rate = float(y[mask_b].mean())
        rates.append(rate)
        counts.append(cnt)
        dur_means.append(float(dur[mask_b].mean()))
        result[f"bin_{b}"] = {
            "n": cnt,
            "mean_duration_ms": float(dur[mask_b].mean()),
            "long_view_rate": rate
        }

    result["overall_rate"] = float(y.mean())
    result["rate_min"] = float(min(rates))
    result["rate_max"] = float(max(rates))
    result["rate_spread"] = float(max(rates) - min(rates))

    # correlation between duration and long_view at row level (point-biserial-ish)
    result["pearson_corr_duration_vs_longview"] = float(np.corrcoef(dur, y)[0, 1])

    # also check monotonicity: count sign changes in successive bin rate diffs
    diffs = np.diff(rates)
    signs = np.sign(diffs)
    sign_changes = int(np.sum(signs[1:] != signs[:-1]))
    result["sign_changes_across_bins"] = sign_changes

    return result