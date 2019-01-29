from copy import deepcopy
import math

def extract_frequent_words(ds, wc_fr):
    ds_fr = deepcopy(ds)
    for d in ds_fr:
        x_wc = d.pop('wc')
        for w in wc_fr:
            d['wc_{w}'.format(w=w)] = x_wc.get(w, 0)
    return ds_fr


# produces interation term between features t1 and t2 
def interaction_term(ds, t1, t2):
    ds_it = deepcopy(ds)
    for d in ds_it:
        d[t1+'_times_'+t2] = d.get(t1,0)*d.get(t2,0)
    return ds_it


# rescales range of feature f to range [0,1] 
def min_max_normalization(ds, f):
    ds_n = deepcopy(ds)
    max = 0
    min = 0
    for d in ds_n: 
        if d[f] > max:
            max = d[f]
        elif d[f] < min:
            min = d[f]
    for d in ds_n:
        d[f+'_normalized'] = (d.get(f,0)-min)/(max-min)
    return ds_n


# add additional feature to ds; namely, the log transformation of feature f
def log_transformation(ds,f):
    ds_log = deepcopy(ds)
    for d in ds_log:
        if d.get(f,0) == 0:
            d[f+'_log'] = 0
        else:
            d[f+'_log'] = math.log(d.get(f,0))
    return ds_log


# add additional feature 'comment_wc' which counts the number of words in the 'text_fpp' feature
def comment_word_count(ds):
    ds_cwc = deepcopy(ds)
    for d in ds_cwc:
        d['comment_wc'] = len(d['text_fpp'])  
    return ds_cwc


# add additional feature 'comment_cc' which counts the number of chars in the text_fpp' feature
def comment_char_count(ds):
    ds_ccc = deepcopy(ds)
    for d in ds_ccc:
        d['comment_cc'] = 0
        for w in d['text_fpp']:
            d['comment_cc'] = d['comment_cc'] + len(w)
    return ds_ccc

# compute bag of words for each comment
def word_count_fpp(ds):
    ds_wc_fpp = deepcopy(ds)
    for d in ds_wc_fpp:
        d['wc_fpp'] = {}
        for w in d['text_fpp']:
            d['wc_fpp'][w] = d['wc_fpp'].get(w, 0) + 1
    return ds_wc_fpp


# compute bag of words for dataset 
def word_count_dataset_fpp(ds):
    wc_fpp = {}
    for d in ds:
        for w in d['text_fpp']:
            wc_fpp[w] = w.get(w, 0) + 1
    return wc_fpp

def calculate_TF(ds):
    ds_tf = deepcopy(ds)
    for d in ds_tf:
        d['tf'] = {}
        wc = len(d['wc_fpp'])
        for w in d['wc_fpp']:
            d['tf'][w] = (d['wc_fpp'].get(w,0))/float(wc)
    return ds_tf

def calculate_IDF(ds, wc_fpp):
    ds_idf = deepcopy(ds)
    idf = {}
    N = len(ds_idf)
    for d in ds_idf:
        for w in wc_fpp:
            if w in d['text_fpp']:
                idf[w] = idf.get(w,0)+1
    for w in idf:
        idf[w] = math.log(N/float(idf.get(w,0)))
    return idf
        
def calculate_TFIDF(ds,idf):
    ds_tfidf = deepcopy(ds)
    for d in ds_tfidf:
        d['tfidf'] = {}
        for w in d['wc_fpp']:
            d['tfidf'][w] = (d['tf'].get(w,0))*(idf.get(w,0))
    return ds_tfidf 
