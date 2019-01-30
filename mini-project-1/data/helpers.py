import json
import logging
from functools import wraps
from time import time

import numpy as np
import pandas as pd
from sklearn import metrics

logger = logging.getLogger(__name__)


def timeit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time()
        result = fn(*args, **kwargs)
        end = time()

        if fn.__module__ == 'linear_regression.closed_form':
            fn_name = 'CF'
        elif fn.__module__ == 'linear_regression.gradient_descent':
            fn_name = 'GD'
        else:
            fn_name = 'SK'
        logger.info('[{fn_name}] T: {time}'.format(fn_name=fn_name, time=end - start))

        return result

    return wrapper


def load_dataset():
    with open('data.json') as f:
        ds = json.loads(f.read())
    return ds


def split_dataset(ds):
    return ds[:10000], ds[10000:11000], ds[11000:12000]


def term_frequency_dataset(ds):
    tf = {}
    for d in ds:
        for w in d['text_pp']:
            tf[w] = tf.get(w, 0) + 1
    return tf


def document_frequency_dataset(ds):
    df = {}
    for i, d in enumerate(ds):
        for w in d['text_pp']:
            if w not in df:
                df[w] = set()
            df[w].add(i)
    df = {k: len(v) for k, v in df.items()}
    return df


def most_frequent_words_dataset(tf, cnt):
    tf_sr = sorted(tf.items(), key=lambda x: -x[1])
    tf_sr_ks = list(map(lambda x: x[0], tf_sr[:cnt]))
    return tf_sr_ks


def uncouple_dataset(ds):
    ds_pd = pd.DataFrame.from_dict(ds, dtype=np.float64)
    y = ds_pd.pop('popularity_score')
    x = ds_pd.drop(['text', 'text_pp', 'tf', 'tf_idf'], axis=1)
    return x, y


def calculate_mse(y, y_prd):
    return metrics.regression.mean_squared_error(y, y_prd)
