import numpy as np

from diffusion_ch.diffusion import Diffusion
from diffusion_ch.utils import load_moons, visualize_ranking
from rank import compute_map_and_print

np.random.seed(42)

def sum_groups(values, group_ids):
    group_sum = {}
    for gid, v in zip(group_ids, values):
        group_sum[gid] = group_sum.get(gid, 0) + v
    group_ids, values = zip(*group_sum.items())
    group_ids, values = np.array(group_ids), np.array(values)
    return values, group_ids


def _args2d_to_indices(argsort_inds):
    rows = np.expand_dims(np.arange(argsort_inds.shape[0]), 1)
    return (rows, argsort_inds)


def sort2d(m, reverse=False):
    ids = np.argsort(m)
    ids_inds = _args2d_to_indices(ids)
    scores = m[ids_inds]

    if not reverse:
        ids = np.fliplr(ids)
        scores = np.fliplr(scores)

    return np.array(scores), np.array(ids)


# %%
X, y = load_moons()
Xt = X[:3]
yt = y[:3]
X = X[3:]
y = y[3:]
# X, y = load_circles()
n = X.shape[0]

Xn = X


# %%
diffusion = Diffusion(k=15, truncation_size=500, affinity="euclidean")
diffusion.fit(Xn)

# %% Diffusion

# offine
q_idx = [600, 686+3]
off_scores, off_ranks = diffusion.offline_search(q_idx, agg=True)
visualize_ranking(X, q_idx=q_idx, k_idx=off_ranks, k_scores=off_scores, contour=False)

# online
on_scores, on_ranks = diffusion.online_search(Xt, agg=True)
visualize_ranking(X, q=Xt, k_idx=on_ranks, k_scores=on_scores, contour=False)

# online
# Xt = np.array([[2., 1.]])
# Xto = np.array([[-1.1, 1.2],
#                [1, -0.25]])
Xto = np.array([[0.4, 0.8],
               [1, -0.50]])
on_scores, on_ranks = diffusion.online_search(Xto, agg=True)
visualize_ranking(X, q=Xto, k_idx=on_ranks, k_scores=on_scores, contour=False)

# online (in db)
q_idx = [5, 6]
Xtdb = X[q_idx]
ond_scores, ond_ranks = diffusion.online_search(Xtdb, agg=True)
visualize_ranking(X, q_idx=q_idx, k_idx=ond_ranks, k_scores=ond_scores, contour=True)

# %%
# yq = y[q_idx][:1]
# yc = y[np.sort(off_ranks)][0]
# gnd = [{"ok": np.argwhere(yc == yq[i])[:, 0]} for i in range(len(yq))]
# compute_map_and_print("oxford5k", off_ranks.T, gnd)
