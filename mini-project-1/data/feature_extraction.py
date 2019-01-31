import math
from copy import deepcopy


def frequent_words(ds, tf_fr):
    ds_fr = deepcopy(ds)
    for d in ds_fr:
        x_tf = d['tf']
        for w in tf_fr:
            d['tf_{w}'.format(w=w)] = x_tf.get(w, 0)
    return ds_fr


def interaction_term(ds, t1, t2):
    ds_it = deepcopy(ds)
    for d in ds_it:
        d[t1 + '_times_' + t2] = int(d[t1]) * int(d[t2])
    return ds_it


def min_max_normalization(ds, fl):
    ds_nr = deepcopy(ds)
    fls = list(map(lambda x: x[fl], ds_nr))
    mx, mn = max(fls), min(fls)
    for d in ds_nr:
        d[fl + '_normalized'] = (d[fl] - mn) / (mx - mn)
    return ds_nr


def log_transformation(ds, fl, bs):
    ds_lg = deepcopy(ds)
    for d in ds_lg:
        d[fl + '_log_' + str(bs)] = 0 if d[fl] == 0 else math.log(d[fl], bs)
    return ds_lg


def power_transformation(ds, fl, pw):
    ds_lg = deepcopy(ds)
    for d in ds_lg:
        d[fl + '_pw_' + str(pw)] = math.pow(d[fl], pw)
    return ds_lg


def word_count(ds):
    ds_wc = deepcopy(ds)
    for d in ds_wc:
        d['wc'] = len(d['text_pp'])
    return ds_wc


def char_count(ds):
    ds_cc = deepcopy(ds)
    for d in ds_cc:
        d['cc'] = sum(map(lambda x: len(x), d['text_pp']))
    return ds_cc


def calculate_tf_idf(ds, df):
    ds_tf_idf = deepcopy(ds)
    for d in ds_tf_idf:
        tf_idf = {}
        for w in d['tf']:
            tf_idf[w] = math.log(1 + d['tf'][w]) * math.log(len(ds) / (1 + df.get(w, 0)))
        d['tf_idf'] = tf_idf
    return ds_tf_idf
