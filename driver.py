import numpy as np

from diffusion_ch.diffusion import Diffusion
from diffusion_ch.utils import load_moons, visualize_ranking

np.random.seed(42)


X, y = load_moons()
# X, y = load_circles()
n = X.shape[0]

Xn = X


diffusion = Diffusion(k=15, truncation_size=500, affinity="euclidean")
diffusion.fit(Xn)

# %% Diffusion

# offine
q_idx = [13, 730]
scores, ranks = diffusion.offline_search(q_idx, agg=True)
visualize_ranking(X, q_idx=q_idx, k_idx=ranks, k_scores=scores, contour=False)

# offine multiple
q_idx = [
    [600],
    [400, 520],
]
scores, ranks = diffusion.offline_search_m(q_idx)
visualize_ranking(X, q_idx=q_idx, k_idx=ranks, k_scores=scores, contour=False)

# online
Xq = [
    [0.83062202, -0.43261387],
    [0.49034901, 0.82519593],
    [1.03343541, -0.48960738],
]

scores, ranks = diffusion.online_search(Xq, agg=True)
visualize_ranking(X, q=Xq, k_idx=ranks, k_scores=scores, contour=False)

# online
Xto = np.array([[0.4, 0.8], [1, -0.50]])
scores, ranks = diffusion.online_search(Xto, agg=True)
visualize_ranking(X, q=Xto, k_idx=ranks, k_scores=scores, contour=False)

# online (in db)
q_idx = [5, 6]
Xtdb = X[q_idx]
scores, ranks = diffusion.online_search(Xtdb, agg=True)
visualize_ranking(X, q_idx=q_idx, k_idx=ranks, k_scores=scores, contour=True)

# online multiple
Xq = [
    np.array([[0.4, 0.8], [1, -0.50]]),
    np.array([[1.75, 0.8], [3, 0.50], [3, 0.55]]),
]
scores, ranks = diffusion.online_search_m(Xq)
visualize_ranking(X, q=Xq, k_idx=ranks, k_scores=scores, contour=False)

# %%
# yq = y[q_idx][:1]
# yc = y[np.sort(off_ranks)][0]
# gnd = [{"ok": np.argwhere(yc == yq[i])[:, 0]} for i in range(len(yq))]
# compute_map_and_print("oxford5k", off_ranks.T, gnd)
