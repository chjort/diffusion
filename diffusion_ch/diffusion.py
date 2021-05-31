import faiss
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from sklearn import preprocessing
from tqdm import tqdm


class Diffusion:
    def __init__(self, k=15, truncation_size=None, affinity="cosine"):
        super(Diffusion, self).__init__()
        self.k = k
        self.truncation_size = truncation_size
        self.affinity = affinity

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

    def _compute_neighborhood_graph(self, X):
        if self.truncation_size is not None:
            k = self.truncation_size  # TODO: Test if truncation here is necessary
        else:
            k = X.shape[0]

        aff, ids = self.knn_.search(X, k)
        return aff, ids

    def _compute_affinity_matrix(self, sims, ids, gamma=3):
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
        sims = sims ** gamma
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

    def _compute_inverse_laplacian(self, neighborhood_ids, laplacian, f0):
        n = laplacian.shape[0]

        # @delayed
        def laplace_inv_col(node_ids, laplacian, f0):
            L_trunc = laplacian[node_ids][:, node_ids]
            c_i, _ = linalg.cg(L_trunc, f0, tol=1e-6, maxiter=20)
            return c_i

        gen = (
            laplace_inv_col(neighborhood_ids[i], laplacian, f0) for i in tqdm(range(n))
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
        X = np.array(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(
                "`features` must have rank 2. Found rank {}.".format(X.ndim)
            )

        self._build_index(X)
        aff, ids = self._compute_neighborhood_graph(X)
        A = self._compute_affinity_matrix(aff, ids, gamma=3)
        D = self._compute_degree_matrix(A)
        S = D.dot(A).dot(D)
        L = self._compute_laplacian_matrix(S)

        f0 = np.zeros(self.truncation_size)
        f0[0] = 1
        L_inv = self._compute_inverse_laplacian(ids, L, f0)
        L_inv = preprocessing.normalize(L_inv, norm="l2", axis=1)
        self.l_inv_ = L_inv

        return self
