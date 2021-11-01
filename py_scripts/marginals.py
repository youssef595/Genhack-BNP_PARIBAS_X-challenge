
import numpy as np
import pandas as pd


def Anderson_Darling(gen_data, real_data):

    # vars
    _, d = gen_data.shape
    L_m = 0

    # compute sum
    for ticker in range(d):
        X = gen_data[:, ticker]
        X_real = real_data[:, ticker]

        L_m += W(X, X_real)

    # additional op
    L_m /= d

    return L_m


def W(X, X_real):

    # vars
    n, = X.shape
    sum_w = 0

    # compute sum
    for i in range(1, n + 1):
        log_u_i = np.log(u(X, X_real, i - 1))
        log_u_n = np.log(u(X, X_real, n - i))

        sum_w += (2 * i - 1) * (log_u_i + log_u_n)

    # additional op
    sum_w /= n
    sum_w -= n

    return sum_w


def u(X, X_real, i):

    # vars
    n, = X.shape
    sum_u = 0

    # create ordered X (synthetic)
    X_order = np.sort(X)

    # compute sum
    for j in range(n):
        sum_u += (X_real[j] <= X_order[i]) + 1

    # additional op
    sum_u /= n + 2

    return sum_u


if __name__ == "__main__":
    # stocks name
    STOCK_NAMES = [i for i in range(4)]

    # load data
    real_data = pd.read_csv(
        # path to data
        "Data/train.csv",
        # use 1st col as index
        index_col=0,
        # name of cols
        header=None,
        names=STOCK_NAMES
    )

    n, d = real_data.shape

    # gen data
    generated_data = np.random.rand(n, d)

    print(Anderson_Darling(generated_data, real_data.values))
