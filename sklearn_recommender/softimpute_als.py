import numpy as np
from numpy.linalg import svd

from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot

from .base import BaseRecommender, csr_center_data


class SoftImputer(BaseRecommender):
    def __init__(self, fm_decoder, alpha=1, n_components=10,
                 max_iter=100,
                 random_state=None):
        BaseRecommender.__init__(self, fm_decoder)
        self.alpha = alpha
        self.fm_decoder = fm_decoder
        self.max_iter = max_iter
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = self.fm_decoder.fm_to_csr(X, y)

        X, self.global_mean_, \
        self.sample_mean_, \
        self.feature_mean_ = csr_center_data(X)

        n_samples, n_features = X.shape
        X = check_array(X, accept_sparse='coo')

        random_state = check_random_state(self.random_state)

        D = np.ones(self.n_components)
        U = random_state.randn(n_samples, self.n_components)
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        V = np.zeros((n_features, self.n_components))

        ii = 0
        while ii < self.max_iter:
            ii += 1
            V, D = sparse_plus_low_rank_ridge(X, U, V, D, self.alpha)
            U, D = sparse_plus_low_rank_ridge(X.T, V, U, D, self.alpha)

        X_proj = X.copy()
        row, col = X.row, X.col
        low_rank = np.sum((U[row] * V[col]) * D[np.newaxis, :] ** 2, axis=1)
        X_proj.data -= low_rank

        M = safe_sparse_dot(X_proj, V)

        self.U, D, R = svd(M, full_matrices=False)
        self.D = np.maximum(D - self.alpha, 0)
        self.V = V.dot(R.T)

        return self

    def _predict_quadratic(self, X_csr, samples, features):
        X_csr.data += np.sum((self.U[samples] * self.V[features])
                      * self.D[np.newaxis, :], axis=1)


def sparse_plus_low_rank_ridge(X, U, V, D, alpha):
    D2 = D ** 2

    X_proj = X.copy()
    row, col = X.row, X.col
    low_rank = np.sum((U[row] * V[col]) * D2[np.newaxis, :], axis=1)
    X_proj.data -= low_rank

    B = safe_sparse_dot(U.T, X_proj) + V.T * D2[:, np.newaxis]
    B *= (D / (D ** 2 + alpha))[:, np.newaxis]

    _, D, V = svd(D[:, np.newaxis] * B, full_matrices=False)
    V = V.T
    D = np.sqrt(D)
    return V, D