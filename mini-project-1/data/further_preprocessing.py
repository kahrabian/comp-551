# Additional text preprocessing for extra features

from copy import deepcopy
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# strip punctuation
def strip_punctuation(ds):
    ds_sp = deepcopy(ds)
    punctuation_list = string.punctuation+'\n'
    for d in ds_sp:
        for i in range(len(d['text_fpp'])):
            for c in punctuation_list:
                d['text_fpp'][i] = d['text_fpp'][i].replace(c,"")
    return ds_sp

# remove all stopwords
def remove_stopwords(ds):
    ds_rs = deepcopy(ds)
    sw_list = list(stopwords.words('english'))
    
    for i in range(len(sw_list)):
        sw_list[i] = sw_list[i].replace("'","")
    
    for d in ds_rs:
        filtered_list = []
        for w in d['text_fpp']:
            if (w not in sw_list) and w:
                filtered_list.append(w)
        d['text_fpp'] =  filtered_list
    return ds_rs

# stem the words
def stem(ds):
    ds_stem = deepcopy(ds)
    stemmer = SnowballStemmer("english")
    for d in ds_stem:
        for i in range(len(d['text_fpp'])):
            d['text_fpp'][i] = stemmer.stem(d['text_fpp'][i])
    return ds_stem

def further_preprocess(ds):
    ds_fpp = deepcopy(ds)
    for d in ds_fpp:
        d['text_fpp'] = d['text_pp']
    ds_np = strip_punctuation(ds_fpp)
    ds_np_rs = remove_stopwords(ds_np)
    ds_np_rs_stem = stem(ds_np_rs)
    return ds_np_rs_stem