import logging
from datetime import datetime

from data.feature_extraction import extract_frequent_words
from data.helpers import load_dataset, split_dataset, word_count_dataset, frequent_words_dataset, uncouple_dataset, \
    calculate_mse, timeit
from data.preprocessing import preprocess

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

    sklearn_lr(x_tr, y_tr, x_cv, y_cv)
    closed_from_lr(x_tr, y_tr, x_cv, y_cv)
    gradient_descent_lr(x_tr, y_tr, x_cv, y_cv)


if __name__ == '__main__':
    main()
