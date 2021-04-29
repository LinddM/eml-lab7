import numpy as np

def cost(Y, X, T):
    return(((X @ T.T - Y) ** 2) * ((Y != 0) * 1)).sum()


def gradient(Y, X, T):
    R = (Y != 0) * 1
    hip_error = (X @ T.T - Y) * R

    return (
        hip_error @ T,
        hip_error.T @ X,
    )


def adam(
    Y, Xo, To,
    fun, jac,
    alpha=0.001, beta1=0.9, beta2=0.999, epsilon=0.0000001,
    max_iter=1000
):

    xm = np.zeros(Xo.shape)
    tm = np.zeros(To.shape)

    xv = np.zeros(Xo.shape)
    tv = np.zeros(To.shape)

    X, T, t = Xo.astype(float), To, 0.0

    while t < max_iter:
        t += 1.0

        xg, tg = jac(Y, X, T)

        # print(f'{t} \t loss={fun(Y, X, T).item():,.2f}')

        xm = beta1 * xm + (1.0 - beta1) * xg
        tm = beta1 * tm + (1.0 - beta1) * tg

        xv = beta2 * xv + (1.0 - beta2) * xg * xg
        tv = beta2 * tv + (1.0 - beta2) * tg * tg

        xmh = xm / (1.0 - beta1 ** t)
        tmh = tm / (1.0 - beta1 ** t)

        xvh = xv / (1.0 - beta2 ** t)
        tvh = tv / (1.0 - beta2 ** t)

        X -= alpha * xmh / (np.sqrt(xvh) + epsilon)
        T -= alpha * tmh / (np.sqrt(tvh) + epsilon)

    return X, T
