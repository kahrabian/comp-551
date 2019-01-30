import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from data import preprocessing
from data.feature_extraction import frequent_words, interaction_term, min_max_normalization, log_transformation, \
    word_count, char_count, calculate_tf_idf
from data.helpers import load_dataset, split_dataset, term_frequency_dataset, most_frequent_words_dataset, timeit, \
    calculate_mse, document_frequency_dataset

logger = logging.getLogger(__name__)


def setup_logger():
    log_path = datetime.now().strftime('./logs/%Y-%m-%d-%H-%M-%S.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())


def sklearn_lr(x_tr, y_tr, x_cv, y_cv):
    from sklearn.linear_model import LinearRegression

    lr_cf = LinearRegression()
    timeit(lr_cf.fit)(x_tr.values, y_tr.values)

    mse = calculate_mse(y_tr.values, lr_cf.predict(x_tr))
    logger.info('[SK] TR MSE: {mse}'.format(mse=mse))

    mse = calculate_mse(y_cv.values, lr_cf.predict(x_cv))
    logger.info('[SK] CV MSE: {mse}'.format(mse=mse))


def closed_from_lr(x_tr, y_tr, x_cv, y_cv):
    from linear_regression.closed_form import ClosedFormLinearRegression

    lr_cf = ClosedFormLinearRegression()
    lr_cf.fit(x_tr.values, y_tr.values)

    mse = calculate_mse(y_tr.values, lr_cf.predict(x_tr))
    logger.info('[CF] TR MSE: {mse}'.format(mse=mse))

    mse = calculate_mse(y_cv.values, lr_cf.predict(x_cv))
    logger.info('[CF] CV MSE: {mse}'.format(mse=mse))


def gradient_descent_lr(x_tr, y_tr, x_cv, y_cv):
    from linear_regression.gradient_descent import GradientDescentLinearRegression

    lr_cf = GradientDescentLinearRegression(beta=1e-6, nu=1e-6, eps=1e-6)
    lr_cf.fit(x_tr.values, y_tr.values)
    y_prd = lr_cf.predict(x_cv)

    mse = calculate_mse(y_cv.values, y_prd)
    logger.info('[GD] CV MSE: {mse}'.format(mse=mse))


def preprocess(ds):
    ds_pp = deepcopy(ds)
    for d in ds_pp:
        d['text_pp'] = deepcopy(d['text'])
    ds_pp = preprocessing.lowercase(ds_pp)
    ds_pp = preprocessing.tokenize(ds_pp)
    ds_pp = preprocessing.strip_punctuation(ds_pp)
    ds_pp = preprocessing.remove_stopwords(ds_pp)
    ds_pp = preprocessing.stem(ds_pp)
    ds_pp = preprocessing.lemmatize(ds_pp)
    ds_pp = preprocessing.term_frequency(ds_pp)
    return ds_pp


def extract_features(ds, df, fw):
    ds_ef = deepcopy(ds)
    ds_ef = frequent_words(ds_ef, fw)
    ds_ef = interaction_term(ds_ef, 'is_root', 'controversiality')
    ds_ef = min_max_normalization(ds_ef, 'children')
    ds_ef = log_transformation(ds_ef, 'children')
    ds_ef = word_count(ds_ef)
    ds_ef = char_count(ds_ef)
    ds_ef = calculate_tf_idf(ds_ef, df)
    return ds_ef


def prepare_dataset(ds):
    ds_pd = pd.DataFrame.from_dict(ds, dtype=np.float64)
    y = ds_pd.pop('popularity_score')
    x = ds_pd.drop(['text', 'text_pp', 'tf', 'tf_idf'], axis=1)
    return x, y


def main():
    setup_logger()

    ds = load_dataset()
    ds_pp = preprocess(ds)
    tr, cv, ts = split_dataset(ds_pp)

    tf = term_frequency_dataset(tr)
    df = document_frequency_dataset(tr)
    fw = most_frequent_words_dataset(tf, 160)

    tr, cv, ts = list(map(lambda x: extract_features(x, df, fw), (tr, cv, ts)))
    (x_tr, y_tr), (x_cv, y_cv), (_, _) = list(map(lambda x: prepare_dataset(x), (tr, cv, ts)))

    sklearn_lr(x_tr, y_tr, x_cv, y_cv)
    closed_from_lr(x_tr, y_tr, x_cv, y_cv)
    gradient_descent_lr(x_tr, y_tr, x_cv, y_cv)


if __name__ == '__main__':
    main()
