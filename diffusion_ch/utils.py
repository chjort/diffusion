import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

np.random.seed(42)


def load_circles():
    n = 500
    X, y = datasets.make_circles(
        n_samples=n, noise=0.05, random_state=42, factor=0.6, shuffle=False
    )

    return X, y


def load_moons():
    n = 500
    X, y = datasets.make_moons(n_samples=n, noise=0.07, random_state=42, shuffle=False)

    x0 = X[y == 0]
    X[:, 0][y == 0] -= 0.25

    x0[:, 0] += 2.25
    y0 = np.zeros(x0.shape[0], dtype=y.dtype)
    X = np.concatenate([X, x0], axis=0)
    y = np.concatenate([y, y0], axis=0)

    return X, y


def scatter2d(x, return_axes=False, **kwargs):
    xax, yax = x[:, 0], x[:, 1]
    plt.scatter(xax, yax, **kwargs)
    if return_axes:
        return xax, yax


def visualize_ranking(X, q_idx=None, k_idx=None, k_scores=None, contour=False):
    scatter2d(X, c="#028ae6")

    for i, (k_idx_i, k_scores_i) in enumerate(zip(k_idx, k_scores)):
        Xk = X[k_idx_i]
        xax, yax = scatter2d(
            Xk,
            c=k_scores_i,
            edgecolors="k",
            s=50,
            linewidths=0.5,
            return_axes=True,
        )
        if i == 0:
            plt.colorbar()
        if contour:
            plt.tricontour(xax, yax, k_scores_i)
            # plt.tricontourf(xax, yax, k_scores_i)

    if q_idx is not None:
        Xq = X[q_idx]
        scatter2d(Xq, c="C3")

    plt.show()
