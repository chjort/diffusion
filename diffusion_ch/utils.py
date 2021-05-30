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


def _get_indices_mask_like(mask, indices):
    idx_mask = np.zeros_like(mask, dtype=bool)
    idx_mask[indices] = True
    return idx_mask


def plot_scatter_2d(
    X,
    y,
    q_idx=None,
    k_idx=None,
    class_color=True,
    color="#028ae6",
    alpha=None,
    k_scores=None,
):
    mask = np.ones(X.shape[0], dtype=bool)

    if q_idx is not None:
        q_mask = _get_indices_mask_like(mask, q_idx)
        mask = np.logical_and(mask, np.logical_not(q_mask))
    else:
        q_mask = None

    if k_idx is not None:
        k_masks = []
        for k_idx_i in k_idx:
            k_mask = _get_indices_mask_like(mask, k_idx_i)
            k_masks.append(k_mask)
            mask = np.logical_and(mask, np.logical_not(k_mask))
    else:
        k_masks = None

    X_data = X[mask]

    if class_color:
        y = y[mask]
        for c in np.unique(y):
            xc = X_data[y == c]
            plt.scatter(xc[:, 0], xc[:, 1], label="c" + str(c), alpha=alpha, s=45)
        plt.legend()
    else:
        plt.scatter(X_data[:, 0], X_data[:, 1], c=color, alpha=alpha, s=45)

    if k_masks is not None:
        for i in range(len(k_masks)):
            k_mask = k_masks[i]
            X_k = X[k_mask]

            if k_scores is not None:
                k_scores_i = k_scores[i]
                plt.scatter(
                    X_k[:, 0],
                    X_k[:, 1],
                    c=k_scores_i,
                    # cmap="viridis",
                    edgecolors="b",
                    s=50,
                    linewidths=0.5,
                )
                plt.colorbar()
                # plt.contour((X_k[:, 0], X_k[:, 1]), k_scores)
            else:
                plt.scatter(
                    X_k[:, 0],
                    X_k[:, 1],
                    c="#73fc03",
                    edgecolors="b",
                    s=50,
                    linewidths=0.5,
                )

    if q_mask is not None:
        X_q = X[q_mask]
        plt.scatter(X_q[:, 0], X_q[:, 1], c="C3", edgecolors="b", s=60, linewidths=0.5)

    plt.show()
