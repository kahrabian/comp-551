from copy import deepcopy


def extract_frequent_words(ds, wc_fr):
    ds_fr = deepcopy(ds)
    for d in ds_fr:
        x_wc = d.pop('wc')
        for w in wc_fr:
            d['wc_{w}'.format(w=w)] = x_wc.get(w, 0)
    return ds_fr
