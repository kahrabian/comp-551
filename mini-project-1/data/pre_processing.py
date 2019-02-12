import logging
import string
from copy import deepcopy

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

logger = logging.getLogger(__name__)


def lowercase(ds):
    logger.info('[PP] lowercase')
    ds_lc = deepcopy(ds)
    for d in ds_lc:
        d['text_pp'] = d['text_pp'].lower()
    return ds_lc


def tokenize(ds):
    logger.info('[PP] tokenize')
    ds_tk = deepcopy(ds)
    for d in ds_tk:
        d['text_pp'] = d['text_pp'].split(' ')
    return ds_tk


def term_frequency(ds):
    logger.info('[PP] term_frequency')
    ds_tf = deepcopy(ds)
    for d in ds_tf:
        tf = {}
        for w in d['text_pp']:
            tf[w] = tf.get(w, 0) + 1
        d['tf'] = tf
    return ds_tf


def strip_punctuation(ds):
    logger.info('[PP] strip_punctuation')
    ds_sp = deepcopy(ds)
    ps = string.punctuation + '\n'
    for d in ds_sp:
        for i, _ in enumerate(d['text_pp']):
            for p in ps:
                d['text_pp'][i] = d['text_pp'][i].replace(p, '')
    return ds_sp


def remove_stopwords(ds):
    logger.info('[PP] remove_stopwords')
    ds_rs = deepcopy(ds)
    sw = stopwords.words('english')
    for i, w in enumerate(sw):
        sw[i] = w.replace('\'', '')
    for d in ds_rs:
        d['text_pp'] = list(filter(lambda x: x not in sw and x != '', d['text_pp']))
    return ds_rs


def stem(ds):
    logger.info('[PP] stem')
    ds_ss = deepcopy(ds)
    ss = SnowballStemmer('english')
    for d in ds_ss:
        for i, w in enumerate(d['text_pp']):
            d['text_pp'][i] = ss.stem(w)
    return ds_ss


def lemmatize(ds):
    logger.info('[PP] lemmatize')
    ds_ss = deepcopy(ds)
    wl = WordNetLemmatizer()
    for d in ds_ss:
        for i, w in enumerate(d['text_pp']):
            d['text_pp'][i] = wl.lemmatize(w)
    return ds_ss
