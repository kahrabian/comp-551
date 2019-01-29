from copy import deepcopy


def lowercase(ds):
    ds_lc = deepcopy(ds)
    for d in ds_lc:
        d['text_pp'] = d['text_pp'].lower()
    return ds_lc


def tokenize(ds):
    ds_tk = deepcopy(ds)
    for d in ds_tk:
        d['text_pp'] = d['text_pp'].split(' ')
    return ds_tk


def word_count(ds):
    ds_wc = deepcopy(ds)
    for d in ds_wc:
        wc = {}
        for w in d['text_pp']:
            wc[w] = wc.get(w, 0) + 1
        d['wc'] = wc
    return ds_wc


def preprocess(ds):
    ds_pp = deepcopy(ds)
    for d in ds_pp:
        d['text_pp'] = d['text']
    ds_lc = lowercase(ds_pp)
    ds_lc_tk = tokenize(ds_lc)
    ds_lc_tk_sc = word_count(ds_lc_tk)
    return ds_lc_tk_sc
