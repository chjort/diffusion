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

# q_idx = [333, 600]
# q_idx = [600, 686]
q_idx = [13 + 3, 686 + 3]
# q_idx = [333, 14]  # circles
k_idx = None

# %%
diffusion = Diffusion(k=15, truncation_size=500, affinity="euclidean")
diffusion.fit(Xn)

# %% Diffusion
L_inv = diffusion.l_inv_
# c_mask[q_idx] = False

# q = np.array(L_inv[q_idx].todense())  # .sum(axis=0, keepdims=True)
# c = np.array(L_inv.todense())

# f_opt = np.matmul(q, np.transpose(c))
# scores, ranks = sort2d(f_opt)
scores, ranks = diffusion.offline_search(q_idx)
ranks.shape

scores
q_idx
ranks



# %%
# c_mask = np.ones(n, dtype=bool)
# c_mask[q_idx] = False

yq = y[q_idx]
yc = y#[c_mask]

# %% Multi-query
# q = q.sum(axis=0, keepdims=True)
# f_opt = np.matmul(q, np.transpose(c))
# scores_m, ranks_m = sort2d(f_opt)

# yq = yq[:1]
# scores = scores_m
# ranks = ranks_m

# %%
gnd = [{"ok": np.argwhere(yc == yq[i])[:, 0]} for i in range(len(yq))]
compute_map_and_print("oxford5k", ranks.T, gnd)

# %%
plot_k = 150
k_idx = ranks[:, :plot_k]
k_scores = scores[:, :plot_k]

# %%
# k_idx_m = ranks_m[:, :plot_k]
# k_scores_m = scores_m[:, :plot_k]

#%%
idx_score_agg = {}
for indices, scs in zip(k_idx, k_scores):
    for idx, sc in zip(indices, scs):
        if idx in idx_score_agg:
            idx_score_agg[idx] += sc
        else:
            idx_score_agg[idx] = sc
k_idx_post = k_idx[:1]
k_scores_post = np.array([[idx_score_agg[idx] for idx in k_idx[0]]])

# k_idx = k_idx_post
# k_scores = k_scores_post

#%% Offline search
# f_opt, ranks = diffusion.offline_search(q_idx)

#%%
visualize_ranking(X, q_idx=q_idx, k_idx=k_idx, k_scores=k_scores, contour=False)
# visualize_ranking(X, q_idx=q_idx, k_idx=k_idx_m, k_scores=k_scores_m, contour=False)
# visualize_ranking(X, q_idx=q_idx, k_idx=k_idx_post, k_scores=k_scores_post, contour=False)

#%% Online search
k_scores, k_idx = diffusion.online_search(Xt)
# k_scores, k_idx = diffusion.online_search_d(Xt)
k_scores = k_scores[-1:, :plot_k]  # NOTE: Last row
k_idx = k_idx[-1:, :plot_k]  # NOTE: Last row
visualize_ranking(X, q_idx=None, k_idx=k_idx, k_scores=k_scores, contour=False)
