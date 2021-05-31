import numpy as np

from diffusion_ch.diffusion import Diffusion
from diffusion_ch.utils import load_moons, visualize_ranking
from rank import compute_map_and_print

np.random.seed(42)


def sort2d(m, reverse=False):
    ids = np.argsort(m)
    scores = np.sort(m)

    if not reverse:
        ids = np.fliplr(ids)
        scores = np.fliplr(scores)

    return np.array(scores), np.array(ids)


# %%
X, y = load_moons()
# X, y = load_circles()
n = X.shape[0]

Xn = X

# q_idx = [333, 600]
q_idx = [13, 686]
# q_idx = [333, 14]  # circles
k_idx = None

# %%
diffusion = Diffusion(k=15, truncation_size=n, affinity="euclidean")
diffusion.fit(Xn)
L_inv = diffusion.l_inv_

# %% Diffusion
c_mask = np.ones(n, dtype=bool)
c_mask[q_idx] = False

q = np.array(L_inv[q_idx].todense())  # .sum(axis=0, keepdims=True)
c = np.array(L_inv[c_mask].todense())

f_opt = np.matmul(q, np.transpose(c))
scores, ranks = sort2d(f_opt)

# %%
yq = y[q_idx]
yc = y[c_mask]

# %% Multi-query
q = q.sum(axis=0, keepdims=True)
f_opt = np.matmul(q, np.transpose(c))
scores, ranks = sort2d(f_opt)
yq = yq[:1]

# %%
gnd = [{"ok": np.argwhere(yc == yq[i])[:, 0]} for i in range(len(yq))]
compute_map_and_print("oxford5k", ranks.T, gnd)

# %%
plot_k = 750
k_idx = ranks[:, :plot_k]
k_scores = scores[:, :plot_k]
visualize_ranking(X, q_idx=q_idx, k_idx=k_idx, k_scores=k_scores, contour=False)
