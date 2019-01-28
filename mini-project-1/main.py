import logging
from datetime import datetime

from sklearn import metrics

from data.feature_extraction import extract_frequent_words
from data.helpers import load_dataset, split_dataset, word_count_dataset, frequent_words_dataset, uncouple_dataset
from data.preprocessing import preprocess


def setup_logger():
    logging.basicConfig(filename=datetime.now().strftime('./logs/%Y-%m-%d-%H-%M-%S.log'), level=logging.INFO)


def mean_squared_error(y, y_prd):
    return metrics.regression.mean_squared_error(y_prd, y.values)


def closed_from_lr(x_tr, y_tr, x_cv, y_cv):
    from linear_regression.closed_form import ClosedFormLinearRegression

    lr_cf = ClosedFormLinearRegression()
    lr_cf.fit(x_tr.values, y_tr.values)
    y_prd = lr_cf.predict(x_cv)

    mse = mean_squared_error(y_cv, y_prd)
    print('The Mean Squared Error is: {mse}'.format(mse=mse))


def main():
    setup_logger()

    ds = load_dataset()
    ds_pp = preprocess(ds)
    tr, cv, ts = split_dataset(ds_pp)

    wc = word_count_dataset(tr)
    wc_fr = frequent_words_dataset(wc, 160)

    tr_fr, cv_fr, ts_fr = list(map(lambda x: extract_frequent_words(x, wc_fr), (tr, cv, ts)))

    x_tr, y_tr = uncouple_dataset(tr_fr)
    x_cv, y_cv = uncouple_dataset(cv_fr)
    # x_ts, y_ts = uncouple_dataset(ts_fr)

    closed_from_lr(x_tr, y_tr, x_cv, y_cv)


if __name__ == '__main__':
    main()
