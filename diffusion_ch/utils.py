import matplotlib.pyplot as plt
import numpy as np
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


def plot_scatter_2d(
    X, y, q_idx=None, k_idx=None, class_color=True, color=None, alpha=None
):
    mask = np.ones(X.shape[0], dtype=bool)

    if q_idx is not None:
        q_mask = np.zeros_like(mask)
        q_mask[q_idx] = True
        mask[q_idx] = False
    else:
        q_mask = None

    if k_idx is not None:
        k_mask = np.zeros_like(mask)
        k_mask[k_idx] = True
        mask[k_idx] = False
    else:
        k_mask = None

    X_data = X[mask]

    if class_color and color is None:
        y = y[mask]
        for c in np.unique(y):
            xc = X_data[y == c]
            plt.scatter(xc[:, 0], xc[:, 1], label="c" + str(c), alpha=alpha, s=45)
        plt.legend()
    else:
        plt.scatter(X_data[:, 0], X_data[:, 1], c="#028ae6", alpha=alpha, s=45)

    if k_mask is not None:
        X_k = X[k_mask]
        plt.scatter(
            X_k[:, 0], X_k[:, 1], c="#73fc03", edgecolors="b", s=50, linewidths=0.5
        )

    if q_mask is not None:
        X_q = X[q_mask]
        plt.scatter(X_q[:, 0], X_q[:, 1], c="C3", edgecolors="b", s=60, linewidths=0.5)

    plt.show()
