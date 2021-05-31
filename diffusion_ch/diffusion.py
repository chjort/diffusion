import faiss
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from sklearn import preprocessing
from tqdm import tqdm


def _args2d_to_indices(argsort_inds):
    rows = np.expand_dims(np.arange(argsort_inds.shape[0]), 1)
    return (rows, argsort_inds)


def sort2d(m, reverse=False):
    """
    Sorts a 2D array along the column axis and returns sorted values
    together with sorted indices.
    """

    ids = np.argsort(m)
    ids_inds = _args2d_to_indices(ids)
    scores = m[ids_inds]

    if not reverse:
        ids = np.fliplr(ids)
        scores = np.fliplr(scores)

    return np.array(scores), np.array(ids)


def sort2d_trunc(m, trunc, reverse=False):
    """
    Sorts a 2D array along the column axis up until `trunc` columns.
    Returns the truncated sorted values together with truncated sorted indices.

    This function is faster than standard sorting followed by truncation.
    """

    parts = np.argpartition(-m, trunc, axis=1)
    parts = parts[:, :trunc]
    parts_inds = _args2d_to_indices(parts)

    ids = np.argsort(-m[parts_inds], axis=1)
    ids_inds = _args2d_to_indices(ids)

    scores = m[parts_inds][ids_inds]
    ids = parts[ids_inds]

    if reverse:
        ids = np.fliplr(ids)
        scores = np.fliplr(scores)

    return scores, ids


# @delayed
def _laplace_inv_col(node_ids, laplacian, f0):
    L_trunc = laplacian[node_ids][:, node_ids]
    c_i, _ = linalg.cg(L_trunc, f0, tol=1e-6, maxiter=20)
    return c_i


def _conform_x(X):
    X = np.array(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("`features` must have rank 2. Found rank {}.".format(X.ndim))
    return X


def _conform_indices(X, ndim=2):
    X = np.array(X, dtype=int)
    if X.ndim != ndim:
        raise ValueError(
            "`features` must have rank {}. Found rank {}.".format(ndim, X.ndim)
        )
    return X


class Diffusion:
    def __init__(self, k=15, truncation_size=None, affinity="cosine", gamma=3):
        super(Diffusion, self).__init__()
        self.k = k
        self.truncation_size = truncation_size
        self.affinity = affinity
        self.gamma = gamma

    def _build_index(self, X):
        d = X.shape[1]
        if self.affinity == "euclidean":
            self.knn_ = faiss.IndexFlatL2(d)
        elif self.affinity == "cosine":
            self.knn_ = faiss.IndexFlatIP(d)
        else:
            raise ValueError(
                "Invalid affinity '{}'. Use 'euclidean' or 'cosine'.".format(
                    self.affinity
                )
            )
        self.knn_.add(X)

    def _knn_search(self, x, k):
        scores, ids = self.knn_.search(x, k)
        scores = 1 - (scores / scores.max(axis=1)[:, None])
        return scores, ids

    def _compute_neighborhood_graph(self, X):
        if self.truncation_size is not None:
            k = self.truncation_size  # TODO: Test if truncation here is necessary
        else:
            k = X.shape[0]

        aff, ids = self._knn_search(X, k)
        return aff, ids

    def _compute_affinity_matrix(self, sims, ids):
        # TODO: Refactor this method!!!
        """Create affinity matrix for the mutual kNN graph of the whole dataset
        Args:
            sims: similarities of kNN
            ids: indexes of kNN
        Returns:
            affinity: affinity matrix
        """
        sims = sims[:, : self.k]
        ids = ids[:, : self.k]

        num = sims.shape[0]
        sims[sims < 0] = 0  # similarity should be non-negative
        sims = sims ** self.gamma
        # vec_ids: feature vectors' ids
        # mut_ids: mutual (reciprocal) nearest neighbors' ids
        # mut_sims: similarites between feature vectors and their mutual nearest neighbors
        vec_ids, mut_ids, mut_sims = [], [], []
        for i in range(num):
            i_nn = ids[i]
            j_nn = ids[i_nn]
            # check reciprocity: i is in j's kNN and j is in i's kNN when i != j
            ismutual = np.isin(j_nn, i).any(axis=1)
            ismutual[0] = False
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
                mut_ids.append(ids[i, ismutual])
                mut_sims.append(sims[i, ismutual])
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix(
            (mut_sims, (vec_ids, mut_ids)), shape=(num, num), dtype=np.float32
        )
        return affinity

    def _compute_degree_matrix(self, affinity_matrix):
        n = affinity_matrix.shape[0]

        ones = np.ones(n)
        degrees = affinity_matrix.dot(ones) + 1e-12
        degrees = degrees ** (-0.5)
        offsets = [0]
        degree_mat = sparse.dia_matrix(
            (degrees, offsets), shape=(n, n), dtype=np.float32
        )
        return degree_mat

    def _compute_laplacian_matrix(self, transition_matrix):
        n = transition_matrix.shape[0]
        ones = np.ones(n)
        offsets = [0]

        alpha = 0.99
        sparse_eye = sparse.dia_matrix((ones, offsets), shape=(n, n), dtype=np.float32)
        laplacian = sparse_eye - alpha * transition_matrix
        return laplacian

    def _compute_inverse_laplacian(self, neighborhood_ids, laplacian):
        n = laplacian.shape[0]

        f0 = np.zeros(self.truncation_size)
        f0[0] = 1

        gen = (
            _laplace_inv_col(neighborhood_ids[i], laplacian, f0) for i in tqdm(range(n))
        )
        # L_inv_cols = Parallel(n_jobs=-1, prefer="threading")(gen)
        L_inv_cols = list(gen)
        L_inv_flat = np.concatenate(L_inv_cols)

        rows = np.repeat(np.arange(n), self.truncation_size)
        cols = neighborhood_ids.reshape(-1)
        L_inv = sparse.csr_matrix(
            (L_inv_flat, (rows, cols)), shape=(n, n), dtype=np.float32
        )

        return L_inv

    def fit(self, X, y=None):
        X = _conform_x(X)

        self._build_index(X)
        aff, ids = self._compute_neighborhood_graph(X)
        A = self._compute_affinity_matrix(aff, ids)
        D = self._compute_degree_matrix(A)
        S = D.dot(A).dot(D)
        L = self._compute_laplacian_matrix(S)

        L_inv = self._compute_inverse_laplacian(ids, L)
        L_inv = preprocessing.normalize(L_inv, norm="l2", axis=1)
        self.l_inv_ = L_inv

        return self

    def offline_search(self, ids):
        try:
            ids = _conform_indices(ids, ndim=1)
        except ValueError:
            ids = _conform_indices(ids, ndim=2)

        f_opt_q = self.l_inv_[ids].toarray()
        f_opt_c = f_opt_q.dot(self.l_inv_.toarray().T)

        # f_opt_c, ranks = sort2d_trunc(f_opt_c, self.truncation_size)
        f_opt_c, ranks = sort2d(f_opt_c)

        if ids.ndim == 1:
            # remove the queries themselves from the ranking
            f_opt_c = f_opt_c[:, 1:]
            ranks = ranks[:, 1:]


        return f_opt_c, ranks

    def initialize(self, X):
        X = _conform_x(X)
        y, nn_ids = self._knn_search(X, k=self.k)
        y = y ** self.gamma

        return y, nn_ids

    def _online_search(self, y, ids):
        n = y.shape[0]

        f_opt = []
        for i in range(n):
            neighbors_i = ids[i]
            L_inv_neighbors = self.l_inv_[neighbors_i].toarray()
            f_opt_i = y[i].dot(L_inv_neighbors)
            f_opt.append(f_opt_i)
        f_opt = np.stack(f_opt, axis=0)
        return f_opt

    def online_search(self, X):
        y, ids = self.initialize(X)
        f_opt = self._online_search(y, ids)
        f_opt, ranks = sort2d_trunc(f_opt, self.truncation_size)
        return f_opt, ranks

    def online_search_d(self, X):
        y, ids = self.initialize(X)
        f_opt = self._online_search(y, ids)
        f_opt = f_opt @ self.l_inv_.toarray()
        f_opt, ranks = sort2d_trunc(f_opt, self.truncation_size)
        return f_opt, ranks
