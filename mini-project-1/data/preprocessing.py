import string
from copy import deepcopy

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer


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


def term_frequency(ds):
    ds_tf = deepcopy(ds)
    for d in ds_tf:
        tf = {}
        for w in d['text_pp']:
            tf[w] = tf.get(w, 0) + 1
        d['tf'] = tf
    return ds_tf


def strip_punctuation(ds):
    ds_sp = deepcopy(ds)
    ps = string.punctuation + '\n'
    for d in ds_sp:
        for i, _ in enumerate(d['text_pp']):
            for p in ps:
                d['text_pp'][i] = d['text_pp'][i].replace(p, '')
    return ds_sp


def remove_stopwords(ds):
    ds_rs = deepcopy(ds)
    sw = stopwords.words('english')
    for i, w in enumerate(sw):
        sw[i] = w.replace('\'', '')
    for d in ds_rs:
        d['text_pp'] = list(filter(lambda x: x not in sw and x != '', d['text_pp']))
    return ds_rs


def stem(ds):
    ds_ss = deepcopy(ds)
    ss = SnowballStemmer('english')
    for d in ds_ss:
        for i, w in enumerate(d['text_pp']):
            d['text_pp'][i] = ss.stem(w)
    return ds_ss


def lemmatize(ds):
    ds_ss = deepcopy(ds)
    wl = WordNetLemmatizer()
    for d in ds_ss:
        for i, w in enumerate(d['text_pp']):
            d['text_pp'][i] = wl.lemmatize(w)
    return ds_ss


def preprocess(ds):
    ds_pp = deepcopy(ds)
    for d in ds_pp:
        d['text_pp'] = deepcopy(d['text'])
    ds_pp = lowercase(ds_pp)
    ds_pp = tokenize(ds_pp)
    ds_pp = strip_punctuation(ds_pp)
    ds_pp = remove_stopwords(ds_pp)
    ds_pp = stem(ds_pp)
    ds_pp = lemmatize(ds_pp)
    ds_pp = term_frequency(ds_pp)
    return ds_pp
