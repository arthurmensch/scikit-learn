import array
from os.path import join

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.externals.joblib import Memory
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot


class SPCARecommender(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=None, n_components=10,
                 alpha=1., l1_ratio=1, algorithm='ridge',
                 debug_folder=None, n_epochs=1,
                 batch_size=10, memory=Memory(cachedir=None)):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.algorithm = algorithm
        self.n_components = n_components
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.memory = memory
        self.debug_folder = debug_folder

    def predict(self, X, n_samples):
        """Prediction from the quadratic term of the factorization machine.

        Returns <Z, XX'>.
        """
        y_hat = np.zeros(X.shape[0])
        _, (samples, features) = fm_format_to_lists(X, y_hat, n_samples)
        # Highly inefficient
        for ix, (i, j) in enumerate(zip(samples, features)):
            y_hat[ix] = (self.global_mean_ + self.sample_effect_[i]
                         + self.feature_effect_[j] +
                         self.code_[i].dot(self.dictionary[j]))
        return y_hat

    def fit(self, X, y, n_samples):
        X_csr = csr_matrix(fm_format_to_coo_matrix(X, y, n_samples))
        interaction = csr_matrix((np.empty_like(X_csr.data),
                                  (X_csr.indices, X_csr.indptr)),
                                 shape=X_csr.shape)
        n_iter = X.shape[0] * self.n_epochs // self.batch_size
        dict_learning = MiniBatchDictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha,
            transform_alpha=self.alpha,
            fit_algorithm=self.algorithm,
            transform_algorithm=self.algorithm,
            l1_ratio=self.l1_ratio,
            batch_size=self.batch_size,
            shuffle=True,
            n_iter=n_iter,
            missing_values=0,
            verbose=10,
            debug_info=True)
        self.code_ = np.zeros((X.shape[0], self.n_components))

        for i in range(self.n_epochs):
            X_csr.data -= interaction
            X_csr, self.global_mean_, self.sample_mean_, self.feature_mean_ = csr_center_data(
                X_csr)
            X_csr.data += interaction.data
            dict_learning.partial_fit(X_csr, deprecated=False)

            self.code_ = dict_learning.transform(X_csr)
            self.dictionary_ = dict_learning.components_
        for i in range(X.shape[0]):
            indices = X.indices[X.indptr[i]:X.indptr[i + 1]]
            interaction.data[X.indptr[i]:X.indptr[i + 1]] = \
                self.code_[i].dot(self.dictionary_[:, indices])

    def transform(self, X):
        X = X.copy()
        X.data[:] = self.global_mean_
        # inter_mean = 0
        for i in range(X.shape[0]):
            if X.indptr[i] < X.indptr[i + 1]:
                X.data[X.indptr[i]:X.indptr[i + 1]] += self.user_mean_[i]
                indices = X.indices[X.indptr[i]:X.indptr[i + 1]]
                interaction = self.code_[i].dot(self.dictionary_[:, indices])
                # inter_mean += interaction.mean() ** 2
                # interaction -= interaction.mean()
                X.data[X.indptr[i]:X.indptr[i + 1]] += interaction
        X.data += self.movie_mean_.take(X.indices, mode='clip')
        return X

    def score(self, X):
        X_pred = self.transform(X)
        if self.debug_folder is not None:
            np.save(join(self.debug_folder, 'X_pred'),
                    X_pred.data)
            np.save(join(self.debug_folder, 'X'),
                    X.data)
        return csr_rmse(X, X_pred)


def make_multinomial_fm_dataset(n_samples, n_features, rank=5, length=50,
                                random_state=None):
    # Inspired by `sklearn.datasets.make_multilabel_classification`
    rng = check_random_state(random_state)

    X_indices = array.array('i')
    X_indptr = array.array('i', [0])

    for i in range(n_samples):
        # pick a non-zero document length by rejection sampling
        n_words = 0
        while n_words == 0:
            n_words = rng.poisson(length)
        # generate a document of length n_words
        words = rng.randint(n_features, size=n_words)
        X_indices.extend(words)
        X_indptr.append(len(X_indices))

    X_data = np.ones(len(X_indices), dtype=np.float64)
    X = sp.csr_matrix((X_data, X_indices, X_indptr),
                      shape=(n_samples, n_features))
    X.sum_duplicates()

    true_w = rng.randn(n_features)
    true_eigv = rng.randn(rank)
    true_P = rng.randn(rank, n_features)

    y = safe_sparse_dot(X, true_w)
    y += ConvexFM().predict_quadratic(X, true_P, true_eigv)
    return X, y


def fm_format_to_lists(X, y, n_samples):
    n_features = X.shape[1] - n_samples
    samples_oh = X[:, :n_samples]
    assert (np.sum(samples_oh == 1, axis=0) == 1)
    assert (np.sum(samples_oh == 0, axis=0) == n_samples - 1)
    features_oh = X[:, n_samples:]
    assert (np.sum(features_oh == 1, axis=0) == 1)
    assert (np.sum(features_oh == 0, axis=0) == n_features - 1)
    _, samples = np.where(samples_oh)
    _, features = np.where(features_oh)
    return (y, (samples, features))


def fm_format_to_coo_matrix(X, y, n_samples):
    n_features = X.shape[1] - n_samples
    y, (samples, features) = fm_format_to_lists(X, y, n_samples)
    X_coo = coo_matrix((y, (samples, features)), shape=(n_samples, n_features))
    return X_coo


def fm_format_to_csr_matrix(X, y, n_samples):
    return csr_matrix(fm_format_to_coo_matrix(X, y, n_samples))


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
