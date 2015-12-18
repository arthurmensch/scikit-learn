from math import sqrt

from nose.tools import assert_greater
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array, check_X_y
import numpy as np
import scipy.sparse as sp


def csr_center_data(X, inplace=False):
    if not inplace:
        X = X.copy()
    w_global = 0

    acc_u = np.zeros(X.shape[0])
    acc_m = np.zeros(X.shape[1])

    n_u = X.getnnz(axis=1)
    n_m = X.getnnz(axis=0)
    n_u[n_u == 0] = 1
    n_m[n_m == 0] = 1
    for i in range(10):
        w_u = X.sum(axis=1).A[:, 0] / n_u
        for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
            X.data[left:right] -= w_u[i]
        w_m = X.sum(axis=0).A[0] / n_m
        X.data -= w_m.take(X.indices, mode='clip')
        acc_u += w_u
        acc_m += w_m

    return X, w_global, acc_u, acc_m


class BaseRecommender(BaseEstimator, RegressorMixin):
    def __init__(self, fm_decoder):
        self.fm_decoder = fm_decoder

    def score(self, X, y, sample_weight=None):
        y_hat = self.predict(X)
        return - sqrt(
            mean_squared_error(y, y_hat, sample_weight=sample_weight))

    def fit(self, X, y):
        X_csr = self.fm_decoder.fm_to_csr(X, y)
        X_csr, self.global_mean_, \
        self.sample_mean_, \
        self.feature_mean_ = csr_center_data(X_csr)

        return self

    def predict(self, X):
        y_hat = np.zeros(X.shape[0])
        X_csr, (samples, features) = self.fm_decoder.fm_to_csr(
            X, y_hat, return_indices=True)
        for i in range(X_csr.shape[0]):
            X_csr.data[X_csr.indptr[i]: X_csr.indptr[i + 1]] += \
                self.sample_mean_[i]
        X_csr.data += self.feature_mean_.take(X_csr.indices, mode='clip')
        self._predict_quadratic(X_csr, samples, features)
        return self.fm_decoder.csr_to_fm(X_csr, return_oh=False,
                                         indices=(samples, features))

    def _predict_quadratic(self, X_csr, samples, features):
        """To be overrided by more complex classes"""
        pass


class FMDecoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_samples=None, n_features=None):
        self.n_samples = n_samples
        self.n_features = n_features

    def fm_to_indices(self, X):
        assert_array_equal(X.indptr,
                           np.arange(0, (X.shape[0] + 1) * 2, 2))
        features = np.maximum(X.indices[1::2], X.indices[::2])
        samples = np.minimum(X.indices[1::2], X.indices[::2])
        assert_greater(features.min(), samples.max())
        existing_n_samples = samples.max() + 1
        existing_n_features = features.max() - features.min() + 1
        assert (existing_n_samples <= self.n_samples)
        assert (existing_n_features <= self.n_features)

        features -= self.n_samples

        return samples, features

    def fm_to_csr(self, X, y=None, return_indices=False):
        X = check_array(X, copy=True)
        (samples, features) = self.fm_to_indices(X)
        if y is None:
            y = np.empty_like(samples)
            y[:] = np.nan
        else:
            y = check_array(y, copy=True)
        X_csr = csr_matrix((y, (samples,
                                features)), shape=(self.n_samples,
                                                   self.n_features))
        X_csr.sort_indices()
        if not return_indices:
            return X_csr
        else:
            return X_csr, (samples, features)

    def csr_to_fm(self, X_csr, return_oh=True, indices=None):
        assert (X_csr.shape == (self.n_samples, self.n_features))

        if indices is None:
            y = X_csr.data
        else:
            if isinstance(indices, tuple):
                indices_samples, indices_features = indices
            elif isinstance(indices, sp.csc_matrix):
                indices_samples, indices_features = self.fm_to_indices(indices)
            y = X_csr[indices_samples, indices_features].A[0]
        if not return_oh:
            return y
        else:
            X = check_array(X_csr, accept_sparse='coo',
                            force_all_finite=False)
            n_rows, n_cols = X_csr.shape
            assert ((n_rows, n_cols) == (self.n_samples, self.n_features))
            if indices is None:
                encoder = OneHotEncoder(n_values=[self.n_samples,
                                                  self.n_features])
                X_ix = np.column_stack([X.row, X.col])
            else:
                assert (np.sorted(indices_samples) == np.sorted(X.row))
                assert (np.sorted(indices_features) == np.sorted(X.col))
                X_ix = np.column_stack([indices_samples, indices_features])
            X_oh = encoder.fit_transform(X_ix)
            return X_oh, y


def array_to_fm_format(X):
    """Converts a dense or sparse array X to factorization machine format.
    If x[i, j] is represented (if X is sparse) or not nan (if dense)
    the output will have a row:
        [one_hot(i), one_hot(j)] -> x[i, j]
    Parameters
    ----------
    X, array-like or sparse
        Input array
    Returns
    -------
    X_one_hot, sparse array, shape (n_x_entries, X.shape[1] + X.shape[2])
        Indices of non-empty values in X, in factorization machine format.
    y: array, shape (n_x_entries,)
        Non-empty values in X.
    """
    X = check_array(X, accept_sparse='coo', force_all_finite=False)
    n_rows, n_cols = X.shape
    encoder = OneHotEncoder(n_values=[n_rows, n_cols])
    if sp.issparse(X):
        y = X.data
        X_ix = np.column_stack([X.row, X.col])
    else:
        ix = np.isfinite(X)
        X_ix = np.column_stack(np.where(ix))
        y = X[ix].ravel()
    X_oh = encoder.fit_transform(X_ix)
    return X_oh, y