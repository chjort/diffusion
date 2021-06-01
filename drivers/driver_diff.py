import numpy as np

from diffusion_ch.diffusion import Diffusion
from diffusion_ch.utils import load_moons, visualize_ranking
from rank import compute_map_and_print

np.random.seed(42)


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

q_idx = [600, 686+3]
k_idx = None

# %%
diffusion = Diffusion(k=15, truncation_size=500, affinity="euclidean")
diffusion.fit(Xn)

# %% Diffusion
scores, ranks = diffusion.offline_search(q_idx, agg=True)

visualize_ranking(X, q_idx=q_idx, k_idx=ranks, k_scores=scores, contour=False)

# q_idx = [[13 + 3, 686 + 3], [600, 686, 606]]
# scores, ranks = diffusion.offline_search_m(q_idx)

# %%
yq = y[q_idx][:1]
yc = y[np.sort(ranks)][0]

# %%
gnd = [{"ok": np.argwhere(yc == yq[i])[:, 0]} for i in range(len(yq))]
compute_map_and_print("oxford5k", ranks.T, gnd)

#%%
plot_k = 150
k_idx = ranks[:, :plot_k]
k_scores = scores[:, :plot_k]
# visualize_ranking(X, q_idx=q_idx, k_idx=k_idx, k_scores=k_scores, contour=False)

#%% Online search
k_scores, k_idx = diffusion.online_search(Xt)
# k_scores, k_idx = diffusion.online_search_d(Xt)
k_scores = k_scores[-1:, :plot_k]  # NOTE: Last row
k_idx = k_idx[-1:, :plot_k]  # NOTE: Last row
# visualize_ranking(X, q_idx=None, k_idx=k_idx, k_scores=k_scores, contour=False)
