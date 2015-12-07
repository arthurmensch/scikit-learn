from os.path import expanduser

import numpy as np
from math import sqrt
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.externals.joblib import Memory
from examples.recommender.convex_fm import array_to_fm_format
from examples.recommender.movielens import fetch_ml_10m
from sklearn.metrics import mean_squared_error


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


class DLRecommender(BaseEstimator, RegressorMixin):
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
        y_hat = np.zeros(X.shape[0])
        y_hat, (samples, features) = fm_format_to_lists(X, y_hat, n_samples)
        X_csr = csr_matrix((y_hat, (samples, features)))
        for i in range(X_csr.shape[0]):
            X_csr.data[X_csr.indptr[i]: X_csr.indptr[i + 1]] += self.sample_mean_[i]
            indices = X_csr.indices[X_csr.indptr[i]:X_csr.indptr[i + 1]]
            X_csr.data[X_csr.indptr[i]:
            X_csr.indptr[i + 1]] += self.code_[i].dot(
                self.dictionary_[:, indices])
        X_csr.data += self.feature_mean_.take(X_csr.indices, mode='clip')

        return X_csr[samples, features].A[0]

    def fit(self, X, y, n_samples):
        X_csr = fm_format_to_csr_matrix(X, y, n_samples)
        X_ref = X_csr.copy()
        interaction = csr_matrix((np.empty_like(X_csr.data),
                                  X_csr.indices, X_csr.indptr),
                                 shape=X_csr.shape)
        n_iter = X_csr.shape[0] * self.n_epochs // self.batch_size
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

        for _ in range(self.n_epochs):
            X_ref.data -= interaction.data
            X_csr, self.global_mean_, self.sample_mean_, self.feature_mean_ = csr_center_data(
                X_ref)
            X_ref.data += interaction.data
            X_csr.data += interaction.data
            dict_learning.partial_fit(X_csr, deprecated=False)

            self.code_ = dict_learning.transform(X_csr)
            self.dictionary_ = dict_learning.components_
            for j in range(X_csr.shape[0]):
                indices = X_csr.indices[X_csr.indptr[j]:X_csr.indptr[j + 1]]
                interaction.data[X_csr.indptr[j]:X_csr.indptr[j + 1]] = \
                    self.code_[j].dot(self.dictionary_[:, indices])


class BaseRecommender(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y, n_samples):
        X_csr = fm_format_to_csr_matrix(X, y, n_samples)
        X_csr, self.global_mean_, \
        self.sample_mean_, \
        self.feature_mean_ = csr_center_data(X_csr)

    def predict(self, X, n_samples):
        y_hat = np.zeros(X.shape[0])
        y_hat, (samples, features) = fm_format_to_lists(X, y_hat, n_samples)
        X_csr = csr_matrix((y_hat, (samples, features)))
        for i in range(X_csr.shape[0]):
            X_csr.data[X_csr.indptr[i]: X_csr.indptr[i + 1]] += self.sample_mean_[i]
        X_csr.data += self.feature_mean_.take(X_csr.indices, mode='clip')

        return X_csr[samples, features].A[0]


def fm_format_to_lists(X, y, n_samples):
    n_features = X.shape[1] - n_samples
    samples_oh = X[:, :n_samples]
    # assert (np.sum(samples_oh == 1, axis=0) == 1)
    # assert (np.sum(samples_oh == 0, axis=0) == n_samples - 1)
    features_oh = X[:, n_samples:]
    # assert (np.sum(features_oh == 1, axis=0) == 1)
    # assert (np.sum(features_oh == 0, axis=0) == n_features - 1)
    samples = samples_oh.indices
    features = features_oh.indices
    return (y, (samples, features))


def fm_format_to_csr_matrix(X, y, n_samples):
    n_features = X.shape[1] - n_samples
    y, (samples, features) = fm_format_to_lists(X, y, n_samples)
    X_csr = csr_matrix((y, (samples, features)), shape=(n_samples, n_features))
    return X_csr


def fit_score_and_dump(estimator, data):
    n_samples, n_features = data.shape
    X, y = array_to_fm_format(data)
    Z = fm_format_to_csr_matrix(X, y, n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
    estimator.fit(X_train, y_train, n_samples)
    y_hat = estimator.predict(X_test, n_samples)
    score = sqrt(mean_squared_error(y_test, y_hat))
    print('RMSE : %.3f' % score)


def main():
    mem = Memory(cachedir=expanduser("~/cache"), verbose=10)
    data = mem.cache(fetch_ml_10m)(expanduser('~/data/own/ml-10M100K'),
                                remove_empty=True)
    fit_score_and_dump(BaseRecommender(), data)
    estimator = DLRecommender(n_components=50,
                              batch_size=10,
                              n_epochs=1,
                              alpha=100,
                              memory=mem,
                              l1_ratio=0.,
                              random_state=0)
    fit_score_and_dump(estimator, data)


if __name__ == '__main__':
    main()