import logging
import math
from copy import deepcopy

logger = logging.getLogger(__name__)


def frequent_words_tf(ds, tf_fr):
    logger.info('[FE] frequent_words_tf')
    ds_fr = deepcopy(ds)
    for d in ds_fr:
        for w in tf_fr:
            d['tf_{w}'.format(w=w)] = d['tf'].get(w, 0)
    return ds_fr


def frequent_words_tf_idf(ds, df, tf_fr):
    logger.info('[FE] frequent_words_tf_idf')
    ds_tf_idf = deepcopy(ds)
    for d in ds_tf_idf:
        for w in tf_fr:
            d['tf_idf_{w}'.format(w=w)] = math.log(1 + d['tf'].get(w, 0)) * math.log(len(ds) / (1 + df.get(w, 0)))
    return ds_tf_idf


def interaction_term(ds, t1, t2):
    logger.info('[FE] interaction_term {t1} {t2}'.format(t1=t1, t2=t2))
    ds_it = deepcopy(ds)
    for d in ds_it:
        d[t1 + '_times_' + t2] = int(d[t1]) * int(d[t2])
    return ds_it


def min_max_normalization(ds, fl):
    logger.info('[FE] min_max_normalization {fl}'.format(fl=fl))
    ds_nr = deepcopy(ds)
    fls = list(map(lambda x: x[fl], ds_nr))
    mx, mn = max(fls), min(fls)
    for d in ds_nr:
        d[fl + '_normalized'] = (d[fl] - mn) / (mx - mn)
    return ds_nr


def log_transformation(ds, fl, bs):
    logger.info('[FE] log_transformation {fl} {bs}'.format(fl=fl, bs=bs))
    ds_lg = deepcopy(ds)
    for d in ds_lg:
        d[fl + '_log_' + str(bs)] = 0 if d[fl] == 0 else math.log(d[fl], bs)
    return ds_lg


def power_transformation(ds, fl, pw):
    logger.info('[FE] power_transformation {fl} {pw}'.format(fl=fl, pw=pw))
    ds_lg = deepcopy(ds)
    for d in ds_lg:
        d[fl + '_pw_' + str(pw)] = math.pow(d[fl], pw)
    return ds_lg


def word_count(ds):
    logger.info('[FE] word_count')
    ds_wc = deepcopy(ds)
    for d in ds_wc:
        d['wc'] = len(d['text_pp'])
    return ds_wc


def char_count(ds):
    logger.info('[FE] char_count')
    ds_cc = deepcopy(ds)
    for d in ds_cc:
        d['cc'] = sum(map(lambda x: len(x), d['text_pp']))
    return ds_cc
