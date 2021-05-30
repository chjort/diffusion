import numpy as np
from sklearn import preprocessing

from diffusion import Diffusion
from diffusion_ch.utils import load_moons, plot_scatter_2d

np.random.seed(42)

#%%
X, y = load_moons()
n = X.shape[0]

Xn = preprocessing.normalize(X, norm="l2", axis=1)

# 8, 10, 17
q_idx = [730, 17]
k_idx = None

# %%
truncation_size = Xn.shape[0]
k = 15

diffusion = Diffusion(Xn)
offline = diffusion.get_offline_results(truncation_size, k)
features = preprocessing.normalize(offline, norm="l2", axis=1)

#%%
c_mask = np.ones(n, dtype=bool)
c_mask[q_idx] = False

q = features[q_idx].todense()
c = features[c_mask].todense()

yq = y[q_idx]
yc = y[c_mask]

scores = np.matmul(q, np.transpose(c))
ranks = np.array(np.argsort(-scores))

# ybin = np.equal(yq[:, None], yc[None, :]).astype(int)
# indices = np.unravel_index(ranks, ybin.shape)
# ybin[indices]

#%%
k_idx = ranks[:, :k].flatten()
plot_scatter_2d(X, y, class_color=False, q_idx=q_idx, k_idx=k_idx, alpha=1.0)
