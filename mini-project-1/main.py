import logging
from datetime import datetime

from data.feature_extraction import extract_frequent_words
from data.helpers import load_dataset, split_dataset, word_count_dataset, frequent_words_dataset, uncouple_dataset
from data.preprocessing import preprocess


def setup_logger():
    logging.basicConfig(filename=datetime.now().strftime('./logs/%Y-%m-%d-%H-%M-%S.log'), level=logging.INFO)


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
    x_ts, y_ts = uncouple_dataset(ts_fr)

    print(x_tr.shape, y_tr.shape)
    print(x_cv.shape, y_cv.shape)
    print(x_ts.shape, y_ts.shape)


if __name__ == '__main__':
    main()
