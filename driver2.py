import numpy as np

from diffusion_ch.diffusion import Diffusion
from diffusion_ch.utils import load_moons, visualize_ranking, load_circles

np.random.seed(42)

PLOT_TOPK = 1000

X, y = load_moons()
# X, y = load_circles()
n = X.shape[0]

# diffusion = Diffusion(k=15, truncation_size=n, affinity="euclidean")
diffusion = Diffusion(k=15, truncation_size=None, affinity="euclidean")
diffusion.fit(X)

#%%
q_idx = [459]
# KNN
Xq = X[q_idx]
scores, ranks = diffusion._knn_search(Xq, n)
scores, ranks = scores[:, :PLOT_TOPK], ranks[:, :PLOT_TOPK]
visualize_ranking(
    X,
    q_idx=q_idx,
    k_idx=ranks,
    k_scores=scores,
    # contour=False,
    contour=True,
    title="KNN",
)

# Diffusion
scores, ranks = diffusion.offline_search(q_idx, agg=False)
scores, ranks = scores[:, :PLOT_TOPK], ranks[:, :PLOT_TOPK]
visualize_ranking(
    X,
    q_idx=np.expand_dims(q_idx, 1),
    k_idx=ranks,
    k_scores=scores,
    contour=True,
    title="Diffusion",
)


#%%
q_idx = [15]
# KNN
Xq = X[q_idx]
scores, ranks = diffusion._knn_search(Xq, n)
scores, ranks = scores[:, :PLOT_TOPK], ranks[:, :PLOT_TOPK]
visualize_ranking(
    X, q_idx=q_idx, k_idx=ranks, k_scores=scores, contour=True, title="KNN"
)

# Diffusion
scores, ranks = diffusion.offline_search(q_idx, agg=False)
scores, ranks = scores[:, :PLOT_TOPK], ranks[:, :PLOT_TOPK]
visualize_ranking(
    X, q_idx=q_idx, k_idx=ranks, k_scores=scores, contour=True, title="Diffusion"
)

#%% Diffusion agg
q_idx = [10, 740]
scores, ranks = diffusion.offline_search(q_idx, agg=True)
scores, ranks = scores[:, :PLOT_TOPK], ranks[:, :PLOT_TOPK]
visualize_ranking(
    X, q_idx=q_idx, k_idx=ranks, k_scores=scores, contour=True, title="Diffusion"
)
