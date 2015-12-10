import datetime
import json
import os
from math import sqrt
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nose.tools import assert_greater
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from examples.recommender.convex_fm import array_to_fm_format
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state, check_array, gen_batches


def fetch_ml_10m(datadir='/volatile/arthur/data/own/ml-10M100K',
                 n_users=None, n_movies=None, remove_empty=True):
    df = pd.read_csv(join(datadir, 'ratings.dat'), sep="::", header=None)
    df.rename(columns={0: 'userId', 1: 'movieId', 2: 'rating', 3: 'timestamp'},
              inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    df.reset_index(drop=True, inplace=True)
    df.set_index(['userId', 'movieId'], inplace=True)
    df.sort_index(level=['userId', 'movieId'], inplace=True)

    ratings = df['rating']

    full_n_users = ratings.index.get_level_values(0).max()
    full_n_movies = ratings.index.get_level_values(1).max()

    X = csr_matrix((ratings, (ratings.index.get_level_values(0) - 1,
                              ratings.index.get_level_values(1) - 1)),
                   shape=(full_n_users, full_n_movies))
    X = X[:n_users, :n_movies]
    if remove_empty:
        rated_movies = (X.getnnz(axis=0) > 2)
        X = X[:, rated_movies]
        rating_users = (X.getnnz(axis=1) > 2)
        X = X[rating_users, :]
    return X

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


def draw_stats(debug_folder):
    residuals = np.load(join(debug_folder, 'residuals.npy'))
    density = np.load(join(debug_folder, 'density.npy'))
    values = np.load(join(debug_folder, 'values.npy'))
    # dictionary = np.load(join(debug_folder, 'dictionary.npy'))
    # code = np.load(join(debug_folder, 'code.npy'))
    probe_score = np.load(join(debug_folder, 'probe_score.npy'))

    fig = plt.figure()
    plt.plot(np.arange(len(residuals)), residuals)
    plt.savefig(join(debug_folder, 'residuals.pdf'))
    plt.close(fig)

    fig = plt.figure()
    plt.plot(np.arange(len(values)), values)
    plt.savefig(join(debug_folder, 'values.pdf'))
    plt.close(fig)

    fig = plt.figure()
    plt.plot(np.arange(len(density)), density)
    plt.savefig(join(debug_folder, 'density.pdf'))
    plt.close(fig)

    fig = plt.figure()
    plt.plot(probe_score[:, 0], probe_score[:, 1:], marker='o')
    plt.savefig(join(debug_folder, 'probe_score.pdf'))
    plt.close(fig)

    # fig = plt.figure(figsize=(10, 10))
    # plt.matshow(dictionary[:, :1000])
    # plt.colorbar()
    # plt.savefig(join(debug_folder, 'dictionary.pdf'))
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10, 10))
    # plt.matshow(code[:1000].reshape((-1, code.shape[1] * 2)))
    # plt.colorbar()
    # plt.savefig(join(debug_folder, 'code.pdf'))
    # plt.close(fig)

    plt.close('all')


class BaseRecommender(BaseEstimator, RegressorMixin):
    def __init__(self, fm_decoder):
        self.fm_decoder = fm_decoder

    def score(self, X, y, sample_weight=None):
        y_hat = self.predict(X)
        return - sqrt(
            mean_squared_error(y, y_hat, sample_weight=sample_weight))

    def fit(self, X, y, **dump_kwargs):
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
        self._predict_quadratic(X_csr)
        return self.fm_decoder.csr_to_fm(X_csr, return_oh=False,
                                         indices=(samples, features))

    def _predict_quadratic(self, X_csr):
        """To be overrided by more complex classes"""
        pass



class DLRecommender(BaseRecommender):
    def __init__(self, fm_decoder=None,
                 random_state=None, n_components=10,
                 alpha=1., l1_ratio=1, algorithm='ridge',
                 n_epochs=1, batch_size=10,
                 memory=Memory(cachedir=None),
                 debug_folder=None,
                 ):
        BaseRecommender.__init__(self, fm_decoder)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.algorithm = algorithm
        self.n_components = n_components
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.memory = memory
        self.debug_folder = debug_folder

    def _predict_quadratic(self, X_csr):
        for i in range(X_csr.shape[0]):
            indices = X_csr.indices[X_csr.indptr[i]:X_csr.indptr[i + 1]]
            X_csr.data[X_csr.indptr[i]:
            X_csr.indptr[i + 1]] += self.code_[i].dot(
                self.dictionary_[:, indices])

    def fit(self, X, y, **dump_kwargs):
        if self.debug_folder is not None:
            self.dump_init()
        X_ref = self.fm_decoder.fm_to_csr(X, y)
        X_csr = X_ref.copy()
        interaction = csr_matrix((np.empty_like(X_csr.data),
                                  X_csr.indices, X_csr.indptr),
                                 shape=X_csr.shape)
        n_iter = X_csr.shape[0] * self.n_epochs // self.batch_size
        random_state = check_random_state(self.random_state)
        dict_init = random_state.randn(self.n_components,
                                       X_csr.shape[1])
        dict_learning = MiniBatchDictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha,
            transform_alpha=self.alpha,
            fit_algorithm=self.algorithm,
            transform_algorithm=self.algorithm,
            dict_init=dict_init,
            l1_ratio=self.l1_ratio,
            batch_size=self.batch_size,
            shuffle=False,
            n_iter=n_iter,
            missing_values=0,
            verbose=10,
            debug_info=self.debug_folder is not None,
            random_state=random_state)
        self.code_ = np.zeros((X.shape[0], self.n_components))

        for i in range(self.n_epochs):
            X_ref.data -= interaction.data
            (X_csr, self.global_mean_,
             self.sample_mean_, self.feature_mean_) = csr_center_data(X_ref)
            X_ref.data += interaction.data
            X_csr.data += interaction.data
            permutation = random_state.permutation(X_csr.shape[0])

            if self.debug_folder is not None:
                batches = gen_batches(X_csr.shape[0],
                                      X_csr.shape[0] // 5)
                last_seen = 0
                for batch in batches:
                    last_seen = max(batch.stop, last_seen)
                    dict_learning.partial_fit(X_csr[permutation[batch]],
                                              deprecated=False)
                    self.dictionary_ = dict_learning.components_
                    self.code_[:last_seen] = dict_learning.transform(
                        X_csr[:last_seen])
                    self.n_iter_ = dict_learning.n_iter_
                    self.dump_inter(debug_dict=dict_learning.debug_info_,
                                    **dump_kwargs)
            else:
                dict_learning.partial_fit(X_csr[permutation], deprecated=False)

            self.dictionary_ = dict_learning.components_
            self.code_ = dict_learning.transform(X_csr)

            for j in range(X_csr.shape[0]):
                indices = X_csr.indices[X_csr.indptr[j]:X_csr.indptr[j + 1]]
                interaction.data[X_csr.indptr[j]:X_csr.indptr[j + 1]] = \
                    self.code_[j].dot(self.dictionary_[:, indices])

        return self

    def dump_init(self):
        result_dict = {'n_components': self.n_components,
                       'l1_ratio': self.l1_ratio,
                       'alpha': self.alpha,
                       'batch_size': self.batch_size}
        with open(join(self.debug_folder, 'results.json'), 'w+') as f:
            json.dump(result_dict, f)

    def dump_inter(self, probe_list=[], debug_dict=None):
        if not hasattr(self, 'probe_score_'):
            self.probe_score_ = []
        probe_score = np.zeros(len(probe_list) + 1)
        probe_score[0] = self.n_iter_
        for i, (X, y) in enumerate(probe_list):
            probe_score[i + 1] = self.score(X, y)
        self.probe_score_.append(probe_score)

        print('Iteration: %i' % probe_score[0])
        for score in probe_score[1:]:
            print('RMSE: %.3f' % score)

        np.save(join(self.debug_folder, 'probe_score'),
                np.array(self.probe_score_))

        with open(join(self.debug_folder, 'results.json'),
                  'r') as f:
            results = json.load(f)
        results['iteration'] = probe_score[0]
        if len(probe_score) > 1:
            results['test_score'] = probe_score[1]
        if len(probe_score) > 2:
            results['train_score'] = probe_score[2]
        with open(join(self.debug_folder, 'results.json'), 'w+') as f:
            json.dump(results, f)

        if debug_dict is not None:
            residuals = debug_dict['residuals']
            density = debug_dict['density']
            values = debug_dict['values']
            np.save(join(self.debug_folder, 'residuals'), residuals)
            np.save(join(self.debug_folder, 'density'), density)
            np.save(join(self.debug_folder, 'values'), values)
        draw_stats(self.debug_folder)


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
        (samples, features) = self.fm_to_indices(X)
        if y is None:
            y = np.empty_like(samples)
            y[:] = np.nan
        X_csr = csr_matrix((y, (samples,
                               features)), shape=(self.n_samples,
                                                        self.n_features))
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
            assert((n_rows, n_cols) == (self.n_samples, self.n_features))
            if indices is None:
                encoder = OneHotEncoder(n_values=[self.n_samples,
                                                  self.n_features])
                X_ix = np.column_stack([X.row, X.col])
            else:
                assert(np.sorted(indices_samples) == np.sorted(X.row))
                assert(np.sorted(indices_features) == np.sorted(X.col))
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


class OHStratifiedShuffleSplit(StratifiedShuffleSplit):
    def __init__(self, fm_decoder, n_iter=5, test_size=0.2, train_size=None,
                 random_state=None):
        self.fm_decoder = fm_decoder
        StratifiedShuffleSplit.__init__(
            self,
            n_iter=n_iter,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)

    def _iter_indices(self, X, y, labels=None):
        samples, features = self.fm_decoder.fm_to_indices(X)
        for train, test in super(OHStratifiedShuffleSplit, self)._iter_indices(
                X, samples):
            yield train, test


def single_run(X, y, estimators, train, test, index):
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    scores = np.zeros(len(estimators))
    for i, estimator in enumerate(estimators):
        estimator.fit(X_train, y_train)
        scores[i] = estimator.score(X_test, y_test)
        print('RMSE %s: %.3f' % (estimator, scores[i]))

    return scores


def main():
    output_dir = expanduser(join('~/output/dl_recommender/',
                                 datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                                  '-%M-%S')))
    os.makedirs(output_dir)

    os.makedirs(join(output_dir, 'non_cv'))

    random_state = check_random_state(0)
    mem = Memory(cachedir=expanduser("~/cache"), verbose=10)
    data = mem.cache(fetch_ml_10m)(expanduser('~/data/own/ml-10M100K'),
                                   remove_empty=True)

    permutation = random_state.permutation(data.shape[0])
    data = data[permutation]

    X, y = array_to_fm_format(data)

    fm_decoder = FMDecoder(n_samples=data.shape[0], n_features=data.shape[1])

    base_estimator = BaseRecommender(fm_decoder)

    dl_rec = DLRecommender(fm_decoder,
                           n_components=50,
                           batch_size=10,
                           n_epochs=5,
                           alpha=100,
                           memory=mem,
                           l1_ratio=0.,
                           # debug_folder=join(output_dir, 'non_cv'),
                           random_state=random_state)

    # dl_cv = GridSearchCV(dl_rec,
    #                      param_grid={'alpha': np.logspace(-3, 3, 7)},
    #                      n_jobs=20,
    #                      error_score='-1000',
    #                      cv=OHStratifiedShuffleSplit(
    #                          fm_decoder,
    #                          n_iter=4, test_size=.1,
    #                          random_state=random_state),
    #                      verbose=10)

    estimators = [base_estimator, dl_rec]

    oh_stratified_shuffle_split = OHStratifiedShuffleSplit(
        fm_decoder,
        n_iter=5,
        test_size=.1, random_state=random_state)

    scores = Parallel(n_jobs=5, verbose=10)(
        delayed(single_run)(X, y, estimators, train, test, index)
        for index, (train, test) in enumerate(
            oh_stratified_shuffle_split.split(X, y)))

    scores = np.array(scores)
    scores = np.mean(scores, axis=0)
    print(scores)

    # dl_cv.fit(X_train, y_train)
    # score = dl_cv.score(X_test, y_test)
    # print('RMSE: %.3f' % score)
    # print(dl_cv.grid_scores_)
    # with open(join(output_dir, 'dl_cv.pkl'), 'wb+') as f:
    #     pickle.dump(dl_cv, f)
    # with open(join(output_dir, 'dl_rec.pkl'), 'wb+') as f:
    #     pickle.dump(dl_rec, f)
    # with open(join(output_dir, 'X.pkl'), 'wb+') as f:
    #     pickle.dump(X, f)
if __name__ == '__main__':
    main()
