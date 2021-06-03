import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

np.random.seed(42)


def load_circles():
    n = 750
    X, y = datasets.make_circles(
        n_samples=n, noise=0.05, random_state=42, factor=0.6, shuffle=True
    )

    X = X.astype(np.float32)
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

    X = X.astype(np.float32)
    return X, y


def _has_ndim(arr, ndim):
    """
    Return True if `arr` is `ndim` dimensional. Also handles the case where first dimension of `arr` is ragged
    """
    return all(np.ndim(ele) == ndim - 1 for ele in arr)


def _1d_to_2d(arr):
    if _has_ndim(arr, 2):
        return arr
    elif _has_ndim(arr, 1):
        arr = [arr]
    else:
        raise ValueError("Array must be 1D or 2D.")

    return arr


def _2d_to_3d(arr):
    if _has_ndim(arr, 3):
        return arr
    elif _has_ndim(arr, 2):
        arr = [arr]
    else:
        raise ValueError("Array must be 2D or 3D.")

    return arr


def scatter2d(x, fig_ax=None, **kwargs):
    x = np.array(x)

    xax, yax = x[:, 0], x[:, 1]
    if fig_ax is not None:
        f = fig_ax.scatter(xax, yax, **kwargs)
    else:
        f = plt.scatter(xax, yax, **kwargs)

    return f, xax, yax


def visualize_ranking(X, q=None, q_idx=None, k_idx=None, k_scores=None, contour=False, title=None):
    if k_idx is not None:
        k_idx = _1d_to_2d(k_idx)
        k_scores = _1d_to_2d(k_scores)

        fig, axes = plt.subplots(1, len(k_idx), figsize=(11, 6))
        if len(k_idx) == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 1)
        axes = np.array([axes])

    for ax in axes:
        scatter2d(X, fig_ax=ax, c="#028ae6")

    if k_idx is not None:
        for i in range(len(k_idx)):
            ax = axes[i]
            if k_scores is not None:
                c = k_scores[i]
            else:
                c = "#73fc03"

            Xk = X[k_idx[i]]
            f, xax, yax = scatter2d(
                Xk,
                fig_ax=ax,
                c=c,
                edgecolors="k",
                s=50,
                linewidths=0.5,
            )
            plt.colorbar(f, ax=ax)
            if contour and k_scores is not None:
                ax.tricontour(xax, yax, k_scores[i])

    if q is not None:
        q = _2d_to_3d(q)
        for qi, ax in zip(q, axes):
            scatter2d(qi, fig_ax=ax, c="C3")

    if q_idx is not None:
        q_idx = _1d_to_2d(q_idx)
        for qid, ax in zip(q_idx, axes):
            Xq = X[qid]
            scatter2d(Xq, fig_ax=ax, c="C3")

    if title is not None:
        plt.suptitle(title, fontsize=14)
    plt.show()
