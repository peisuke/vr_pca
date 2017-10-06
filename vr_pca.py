import numpy as np


def vr_pca(X, m, eta, rate=1e-5):
    n, d = X.shape
    w_t = np.random.rand(d) - 0.5
    w_t = w_t / np.linalg.norm(w_t)

    for s in range(10):
        u_t = X.T.dot(X.dot(w_t)) / n

        w = []
        w.append(w_t)

        for t in range(m):
            i = np.random.randint(n)
            _w = w[t] + eta * (X[i] * (X[i].T.dot(w[t]) - X[i].T.dot(w_t)) + u_t)
            _w = _w / np.linalg.norm(_w)
            w.append(_w)

        d = np.linalg.norm(w_t - w[-1])
        w_t = w[-1]

        if d < rate:
            return w_t
