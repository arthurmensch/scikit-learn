import numpy as np
from scipy.linalg import svd

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot


class softImpute(BaseEstimator):
    def __init__(self, alpha=1, n_components=10,
                 max_iter=100,
                 random_state=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_components = n_components
        self.random_state = random_state


    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        X = check_array(X, accept_sparse='coo')
        random_state = check_random_state(self.random_state)
        D = np.ones(self.n_components)
        U = random_state.randn(n_samples, self.n_components)
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        V = np.zeros(n_features, self.n_components)
        ii = 0
        while ii < self.max_iter:
            ii += 1

            V, D = sparse_plus_low_rank_ridge(self, X, U, V, D, self.alpha)
            U, D = sparse_plus_low_rank_ridge(self, X.T, V, U, D, self.alpha)
        M = safe_sparse_dot(X, V)
        U, D, R = svd(M)
        D = np.maximum(D[:self.n_components] - self.alpha, 0)


def sparse_plus_low_rank_ridge(self, X, U, V, D, alpha):
    D2 = D ** 2

    X_proj = X.copy()
    row, col = X.nonzero()
    low_rank = np.sum(U[row] * V[col] * D2, axis=1)
    X_proj.data -= low_rank

    B = safe_sparse_dot(U.T, X_proj) + V.T * D2[np.newaxis, :]
    B *= D / (D ** 2 + alpha)[np.newaxis, :]

    V, D, _ = svd(B * D[:, np.newaxis])
    D = np.sqrt(D)
    return V, D