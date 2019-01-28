from sklearn import datasets, metrics


def sklearn_lr(ds):
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(ds.data, ds.target)
    y_prd = lr.predict(ds.data)

    return y_prd


def closed_form_lr(ds):
    from linear_regression.closed_form import ClosedFormLinearRegression

    lr = ClosedFormLinearRegression()
    lr.fit(ds.data, ds.target)
    y_prd = lr.predict(ds.data)

    return y_prd


def gradient_descent_lr(ds):
    from linear_regression.gradient_descent import GradientDescentLinearRegression

    lr = GradientDescentLinearRegression(beta=1e-6, nu=5e-9, eps=1e-6)
    lr.fit(ds.data, ds.target)
    y_prd = lr.predict(ds.data)

    return y_prd


def test():
    ds = datasets.load_boston()

    sk_prd = sklearn_lr(ds)
    cf_prd = closed_form_lr(ds)
    gd_prd = gradient_descent_lr(ds)

    mse_sk = metrics.regression.mean_squared_error(sk_prd, ds.target)
    mse_cf = metrics.regression.mean_squared_error(cf_prd, ds.target)
    mse_gd = metrics.regression.mean_squared_error(gd_prd, ds.target)

    assert abs(mse_sk - mse_cf) < 1e-10
    assert abs(mse_sk - mse_gd) < 3


if __name__ == '__main__':
    test()
