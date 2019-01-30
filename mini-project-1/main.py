import logging
from datetime import datetime

from data.feature_extraction import frequent_words, interaction_term, min_max_normalization, log_transformation, \
    word_count, char_count, calculate_tf_idf
from data.helpers import load_dataset, split_dataset, term_frequency_dataset, most_frequent_words_dataset, timeit, \
    uncouple_dataset, calculate_mse, document_frequency_dataset
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

    tf = term_frequency_dataset(tr)
    df = document_frequency_dataset(tr)

    fw = most_frequent_words_dataset(tf, 160)

    tr, cv, ts = list(map(lambda x: frequent_words(x, fw), (tr, cv, ts)))
    tr, cv, ts = list(map(lambda x: interaction_term(x, 'is_root', 'controversiality'), (tr, cv, ts)))
    tr, cv, ts = list(map(lambda x: min_max_normalization(x, 'children'), (tr, cv, ts)))
    tr, cv, ts = list(map(lambda x: log_transformation(x, 'children'), (tr, cv, ts)))
    tr, cv, ts = list(map(lambda x: word_count(x), (tr, cv, ts)))
    tr, cv, ts = list(map(lambda x: char_count(x), (tr, cv, ts)))
    tr, cv, ts = list(map(lambda x: calculate_tf_idf(x, df), (tr, cv, ts)))

    x_tr, y_tr = uncouple_dataset(tr)
    x_cv, y_cv = uncouple_dataset(cv)
    # x_ts, y_ts = uncouple_dataset(ts)

    sklearn_lr(x_tr, y_tr, x_cv, y_cv)
    closed_from_lr(x_tr, y_tr, x_cv, y_cv)
    gradient_descent_lr(x_tr, y_tr, x_cv, y_cv)


if __name__ == '__main__':
    main()
