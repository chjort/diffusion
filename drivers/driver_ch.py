import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from sklearn import preprocessing
from tqdm import tqdm

from diffusion import Diffusion
from diffusion_ch.utils import load_moons, visualize_ranking, load_circles
from rank import compute_map_and_print

np.random.seed(42)


def sort2d(m, reverse=False):
    ids = np.argsort(m)
    scores = np.sort(m)

    if not reverse:
        ids = np.fliplr(ids)
        scores = np.fliplr(scores)

    return np.array(scores), np.array(ids)


#%%
# query_path = "data/query/oxford5k_resnet_glob.npy"
# gallery_path = "data/gallery/oxford5k_resnet_glob.npy"
# dataset = Dataset(query_path, gallery_path)
# queries, gallery = dataset.queries, dataset.gallery
#
# X = np.vstack([queries, gallery])
# n_query = len(queries)
####
# gnd_path = "data/gnd_oxford5k.pkl"
# gnd_name = os.path.splitext(os.path.basename(gnd_path))[0]
# with open(gnd_path, "rb") as f:
#     gnd = pickle.load(f)["gnd"]

#%%
X, y = load_moons()
# X, y = load_circles()
n = X.shape[0]

# Xn = preprocessing.normalize(X, norm="l2", axis=1)
# diffusion = Diffusion(Xn, method="cosine")

Xn = X
diffusion = Diffusion(Xn, method="euclidean")

# q_idx = [333, 600]
q_idx = [13, 686]
# q_idx = [333, 14]  # circles
k_idx = None


# %%
truncation_size = n
k = 15

#%% Construct graph (A)
sims, ids = diffusion.knn.search(diffusion.features, truncation_size)
sims = 1 - (sims / sims.max(axis=1)[:, None])
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

q = np.array(L_inv[q_idx].todense())#.sum(axis=0, keepdims=True)
c = np.array(L_inv[c_mask].todense())

f_opt = np.matmul(q, np.transpose(c))
scores, ranks = sort2d(f_opt)

#%%
yq = y[q_idx]
yc = y[c_mask]

#%% Multi-query
q = q.sum(axis=0, keepdims=True)
f_opt = np.matmul(q, np.transpose(c))
scores, ranks = sort2d(f_opt)
yq = yq[:1]

#%%
gnd = [{"ok": np.argwhere(yc == yq[i])[:, 0]} for i in range(len(yq))]

## KNN scores
ranks_a = ids[q_idx]
scores_a = sims[q_idx]

compute_map_and_print("oxford5k", ranks.T, gnd)
compute_map_and_print("oxford5k", ranks_a.T, gnd)

#%%
# plot_k = 50
# k_idx = ranks_a[:, :plot_k]
# k_scores = scores_a[:, :plot_k]
# visualize_ranking(X, q_idx=q_idx, k_idx=k_idx, k_scores=k_scores, contour=False)

plot_k = 750
k_idx = ranks[:, :plot_k]
k_scores = scores[:, :plot_k]
visualize_ranking(X, q_idx=q_idx, k_idx=k_idx, k_scores=k_scores, contour=True)
