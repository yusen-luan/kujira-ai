def run_probe(arrays, masks, user_feat, video_feat, label):
    from collections import defaultdict

    train_mask = masks['train']
    uids = arrays['user_id'][train_mask]
    vids = arrays['video_id'][train_mask]
    lv = arrays['long_view'][train_mask].astype(float)

    def group_rates(keys):
        sum_ = defaultdict(float)
        cnt_ = defaultdict(int)
        for k, y in zip(keys, lv):
            sum_[k] += y
            cnt_[k] += 1
        return sum_, cnt_

    uad = [user_feat.get(str(u), {}).get('user_active_degree', 'NA') for u in uids]
    sum_u, cnt_u = group_rates(uad)

    vtype = [video_feat.get(str(v), {}).get('video_type', 'NA') for v in vids]
    sum_v, cnt_v = group_rates(vtype)

    def first_tag(v):
        t = video_feat.get(str(v), {}).get('tag', None)
        if not t:
            return 'NA'
        return t.split(',')[0]
    tag = [first_tag(v) for v in vids]
    sum_t, cnt_t = group_rates(tag)

    utype = [video_feat.get(str(v), {}).get('upload_type', 'NA') for v in vids]
    sum_ut, cnt_ut = group_rates(utype)

    def summarize(sum_, cnt_, min_support=200, topk=6):
        rates = {k: sum_[k] / cnt_[k] for k in cnt_ if cnt_[k] >= min_support}
        if not rates:
            return {}
        items = sorted(((k, cnt_[k], sum_[k] / cnt_[k]) for k in rates), key=lambda x: -x[1])[:topk]
        return {
            'n_groups_min_support': len(rates),
            'rate_min': min(rates.values()),
            'rate_max': max(rates.values()),
            'rate_spread': max(rates.values()) - min(rates.values()),
            'top_groups_by_count': [(str(k), int(c), float(r)) for k, c, r in items],
        }

    return {
        'overall_long_view_rate_train': float(lv.mean()),
        'user_active_degree': summarize(sum_u, cnt_u),
        'video_type': summarize(sum_v, cnt_v),
        'tag_first': summarize(sum_t, cnt_t, min_support=100),
        'upload_type': summarize(sum_ut, cnt_ut),
    }