import json

import numpy as np
import pandas as pd


def load_dataset():
    with open('data.json') as f:
        ds = json.loads(f.read())
    return ds


def split_dataset(ds):
    return ds[:10000], ds[10000:11000], ds[11000:12000]


def word_count_dataset(ds):
    wc = {}
    for d in ds:
        for w in d['text_pp']:
            wc[w] = wc.get(w, 0) + 1
    return wc


def frequent_words_dataset(wc, cnt):
    wc_sr = sorted(filter(lambda x: x[0] != '', wc.items()), key=lambda x: -x[1])
    wc_sr_ks = list(map(lambda x: x[0], wc_sr[:cnt]))
    return wc_sr_ks


def uncouple_dataset(ds):
    ds_pd = pd.DataFrame.from_dict(ds, dtype=np.float64)
    y = ds_pd.pop('popularity_score')
    x = ds_pd.drop(['text', 'text_pp'], axis=1)
    return x, y
