import os
import pickle

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from joblib import Parallel, delayed
from sklearn import preprocessing
from tqdm import tqdm

from dataset import Dataset
from diffusion import Diffusion
from diffusion_ch.utils import load_moons, plot_scatter_2d, load_circles
from rank import compute_map_and_print

np.random.seed(42)

#%%
# query_path = "data/query/oxford5k_resnet_glob.npy"
# gallery_path = "data/gallery/oxford5k_resnet_glob.npy"
# dataset = Dataset(query_path, gallery_path)
# queries, gallery = dataset.queries, dataset.gallery
#
# X = np.vstack([queries, gallery])
# n_query = len(queries)

#%%
X, y = load_moons()
# X, y = load_circles()
n = X.shape[0]

# Xn = preprocessing.normalize(X, norm="l2", axis=1)
# diffusion = Diffusion(Xn, method="cosine")

Xn = X
diffusion = Diffusion(Xn, method="euclidean")

# q_idx = [200, 600]
# q_idx = np.arange(n_query)
k_idx = None

q_idx = [333, 14]  # circles

# %%
truncation_size = n
# truncation_size = 1000
# k = 5
k = 15
# k = 50

#%% Construct graph (A)
sims, ids = diffusion.knn.search(diffusion.features, truncation_size)
trunc_ids = ids

#%% Affinity matrix (A)
sims_k, ids_k = sims[:, :k], ids[:, :k]
A = diffusion.get_affinity(sims_k, ids_k, gamma=3)

#%% Degree matrix (D)
ones = np.ones(n)
degrees = A @ ones + 1e-12
degrees = degrees ** (-0.5)
offsets = [0]
D = sparse.dia_matrix((degrees, offsets), shape=(n, n), dtype=np.float32)

#%% Transition matrix (S)
S = D @ A @ D

#%% Laplacian matrix
alpha = 0.99
sparse_eye = sparse.dia_matrix((ones, offsets), shape=(n, n), dtype=np.float32)
L = sparse_eye - alpha * S

#%% Compute inverse laplacian
def compute_inverse_laplacian(neighborhood_ids, laplacian, f0):
    # @delayed
    def laplace_inv_col(node_ids, laplacian, f0):
        L_trunc = laplacian[node_ids][:, node_ids]
        c_i, _ = linalg.cg(L_trunc, f0, tol=1e-6, maxiter=20)
        return c_i

    gen = (laplace_inv_col(neighborhood_ids[i], laplacian, f0) for i in tqdm(range(n)))
    # L_inv_cols = Parallel(n_jobs=-1, prefer="threading")(gen)
    L_inv_cols = list(gen)
    L_inv_flat = np.concatenate(L_inv_cols)

    rows = np.repeat(np.arange(n), truncation_size)
    cols = neighborhood_ids.reshape(-1)
    L_inv = sparse.csr_matrix(
        (L_inv_flat, (rows, cols)), shape=(n, n), dtype=np.float32
    )

    return L_inv


f0 = np.zeros(truncation_size)  # `y` in Iscen et al.
f0[0] = 1

L_inv = compute_inverse_laplacian(trunc_ids, L, f0)
L_inv = preprocessing.normalize(L_inv, norm="l2", axis=1)

#%% Diffusion
c_mask = np.ones(n, dtype=bool)
c_mask[q_idx] = False

q = np.array(L_inv[q_idx].todense())
c = np.array(L_inv[c_mask].todense())

f_opt = np.matmul(q, np.transpose(c))
ranks = np.fliplr(np.argsort(f_opt))

#%%
# gnd_path = "data/gnd_oxford5k.pkl"
# gnd_name = os.path.splitext(os.path.basename(gnd_path))[0]
# with open(gnd_path, "rb") as f:
#     gnd = pickle.load(f)["gnd"]


yq = y[q_idx]
yc = y[c_mask]
gnd = [{"ok": np.argwhere(yc == yq[0])[:, 0]}, {"ok": np.argwhere(yc == yq[1])[:, 0]}]

# ybin = np.equal(yq[:, None], yc[None, :]).astype(int)
# indices = np.unravel_index(ranks, ybin.shape)
# ybin[indices]

ranks_a = ids[q_idx]
compute_map_and_print("oxford5k", ranks.T, gnd)
compute_map_and_print("oxford5k", ranks_a.T, gnd)

#%%
plot_k = 80
k_idx = ranks_a[:, 1 : plot_k + 1].flatten()
plot_scatter_2d(X, y, class_color=False, q_idx=q_idx, k_idx=k_idx, alpha=1.0)
# plot_scatter_2d(Xn, y, class_color=False, q_idx=q_idx, k_idx=k_idx, alpha=1.0)

k_idx = ranks[:, :plot_k].flatten()
plot_scatter_2d(X, y, class_color=False, q_idx=q_idx, k_idx=k_idx, alpha=1.0)
# plot_scatter_2d(Xn, y, class_color=False, q_idx=q_idx, k_idx=k_idx, alpha=1.0)


#%%
scores = np.fliplr(np.sort(f_opt))

ranks
scores
