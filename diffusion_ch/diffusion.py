import faiss
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from sklearn import preprocessing
from tqdm import tqdm


def _sum_groups(values, group_ids):
    """Groups `values` by `group_ids` and sum values in each group"""
    group_sum = {}
    for gid, v in zip(group_ids, values):
        group_sum[gid] = group_sum.get(gid, 0) + v
    group_ids, values = zip(*group_sum.items())
    group_ids, values = np.array(group_ids), np.array(values)
    return values, group_ids


def _args2d_to_indices(argsort_inds):
    rows = np.expand_dims(np.arange(argsort_inds.shape[0]), 1)
    return rows, argsort_inds


def _sort2d(m, reverse=False):
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


def _sort2d_trunc(m, trunc, reverse=False):
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
    l_trunc = laplacian[node_ids][:, node_ids]
    c_i, _ = linalg.cg(l_trunc, f0, tol=1e-6, maxiter=20)
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


def _remove_query_ids_1d(f_opt, ranks, ids):
    not_query = np.isin(ranks, ids, invert=True)
    return f_opt[not_query], ranks[not_query]


def _compute_degree_matrix(affinity_matrix):
    n = affinity_matrix.shape[0]

    ones = np.ones(n)
    degrees = affinity_matrix.dot(ones) + 1e-12
    degrees = degrees ** (-0.5)
    offsets = [0]
    degree_mat = sparse.dia_matrix((degrees, offsets), shape=(n, n), dtype=np.float32)
    return degree_mat


def _compute_laplacian_matrix(transition_matrix):
    n = transition_matrix.shape[0]
    ones = np.ones(n)
    offsets = [0]

    alpha = 0.99
    sparse_eye = sparse.dia_matrix((ones, offsets), shape=(n, n), dtype=np.float32)
    laplacian = sparse_eye - alpha * transition_matrix
    return laplacian


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
            k = self.truncation_size
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
            ismutual[0] = False  # False where i = j
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
                mut_ids.append(i_nn[ismutual])
                mut_sims.append(sims[i, ismutual])
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix(
            (mut_sims, (vec_ids, mut_ids)), shape=(num, num), dtype=np.float32
        )
        return affinity

    def _compute_inverse_laplacian(self, neighborhood_ids, laplacian):
        n = laplacian.shape[0]
        if self.truncation_size is not None:
            n_trunc = self.truncation_size
        else:
            n_trunc = n

        f0 = np.zeros(n_trunc)
        f0[0] = 1

        gen = (
            _laplace_inv_col(neighborhood_ids[i], laplacian, f0) for i in tqdm(range(n))
        )
        # l_inv_cols = Parallel(n_jobs=-1, prefer="threading")(gen)
        l_inv_cols = list(gen)
        l_inv_flat = np.concatenate(l_inv_cols)

        rows = np.repeat(np.arange(n), n_trunc)
        cols = neighborhood_ids.reshape(-1)
        l_inv = sparse.csr_matrix(
            (l_inv_flat, (rows, cols)), shape=(n, n), dtype=np.float32
        )

        return l_inv

    def fit(self, X):
        X = _conform_x(X)

        self._build_index(X)
        aff, ids = self._compute_neighborhood_graph(X)
        A = self._compute_affinity_matrix(aff, ids)
        D = _compute_degree_matrix(A)
        S = D.dot(A).dot(D)
        L = _compute_laplacian_matrix(S)

        l_inv = self._compute_inverse_laplacian(ids, L)
        l_inv = preprocessing.normalize(l_inv, norm="l2", axis=1)
        self.l_inv_ = l_inv

        return self

    def _initialize_offline(self, ids, agg=False):
        ids = _conform_indices(ids, ndim=1)
        c_i = self.l_inv_[ids].toarray()

        if agg:
            c_i = c_i.sum(axis=0, keepdims=True)

        return c_i

    def _diffuse_offline(self, y):
        l_inv = self.l_inv_.toarray().T
        f_opt = y.dot(l_inv)
        return f_opt

    def offline_search_m(self, ids):
        ids = [_conform_indices(ids_i, ndim=1) for ids_i in ids]
        c_qs = np.concatenate([self._initialize_offline(id_, agg=True) for id_ in ids])
        f_opts = self._diffuse_offline(c_qs)
        f_opts, ranks = _sort2d(f_opts)

        f_opts, ranks = zip(
            *[
                _remove_query_ids_1d(f_opt_i, ranks_i, ids_i)
                for f_opt_i, ranks_i, ids_i in zip(f_opts, ranks, ids)
            ]
        )

        try:
            f_opts, ranks = np.stack(f_opts), np.stack(ranks)
        except ValueError:
            # Variable number of neighbors, so we cant stack into a numpy matrix.
            pass

        return f_opts, ranks

    def offline_search(self, ids, agg=False):
        ids = _conform_indices(ids, ndim=1)
        c_q = self._initialize_offline(ids, agg=agg)
        f_opt = self._diffuse_offline(c_q)

        if (
            agg
            or self.truncation_size is None
            or self.truncation_size >= self.l_inv_.shape[0]
        ):
            f_opt, ranks = _sort2d(f_opt)
            f_opt, ranks = _remove_query_ids_1d(f_opt, ranks, ids)
            f_opt, ranks = np.expand_dims(f_opt, 0), np.expand_dims(ranks, 0)
        else:
            f_opt, ranks = _sort2d_trunc(f_opt, self.truncation_size)
            f_opt, ranks = zip(
                *[
                    _remove_query_ids_1d(f_opt_i, ranks_i, ids_i)
                    for f_opt_i, ranks_i, ids_i in zip(f_opt, ranks, ids)
                ]
            )
            f_opt, ranks = np.stack(f_opt), np.stack(ranks)

        return f_opt, ranks

    def _initialize_online(self, X, agg=False):
        X = _conform_x(X)
        y, nn_ids = self._knn_search(X, k=self.k)
        y = y ** self.gamma

        if agg:
            y, nn_ids = _sum_groups(y.flatten(), nn_ids.flatten())
            y, nn_ids = np.expand_dims(y, 0), np.expand_dims(nn_ids, 0)

        return y, nn_ids

    def _diffuse_online(self, y, ids):
        n = y.shape[0]

        f_opt = []
        for i in range(n):
            neighbors_i = ids[i]
            L_inv_neighbors = self.l_inv_[neighbors_i].toarray()
            f_opt_i = y[i].dot(L_inv_neighbors)
            f_opt.append(f_opt_i)
        f_opt = np.stack(f_opt, axis=0)
        return f_opt

    def online_search_m(self, X):
        X = [_conform_x(x_i) for x_i in X]
        ys_ids = [self._initialize_online(x_i, agg=True) for x_i in X]
        f_opts = np.concatenate([self._diffuse_online(y, id_) for y, id_ in ys_ids])
        f_opt, ranks = _sort2d(f_opts)
        return f_opt, ranks

    def online_search(self, X, agg=False):
        y, ids = self._initialize_online(X, agg=agg)
        f_opt = self._diffuse_online(y, ids)

        if (
            agg
            or self.truncation_size is None
            or self.truncation_size >= self.l_inv_.shape[0]
        ):
            f_opt, ranks = _sort2d(f_opt)
        else:
            f_opt, ranks = _sort2d_trunc(f_opt, self.truncation_size)

        return f_opt, ranks
