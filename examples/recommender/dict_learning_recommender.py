import datetime
import json
import os
from os.path import expanduser, join
import numpy as np
from math import sqrt

from multiprocessing import Process
from nose.tools import assert_greater
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.externals.joblib import Memory
from examples.recommender.convex_fm import array_to_fm_format
from examples.recommender.movielens import fetch_ml_10m
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state, check_array, gen_batches
import matplotlib.pyplot as plt


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
    def __init__(self):
        pass

    def score(self, X, y, sample_weight=None):
        y_hat = self.predict(X)
        return - sqrt(mean_squared_error(y, y_hat, sample_weight=sample_weight))

    def fit(self, X, y, **dump_kwargs):
        X_csr = fm_format_to_csr_matrix(X, y)
        X_csr, self.global_mean_, \
        self.sample_mean_, \
        self.feature_mean_ = csr_center_data(X_csr)

    def predict(self, X):
        y_hat = np.zeros(X.shape[0])
        X_csr, (samples, features) = fm_format_to_csr_matrix(X,
                                                             y_hat,
                                                             return_lists=True)
        for i in range(X_csr.shape[0]):
            X_csr.data[X_csr.indptr[i]: X_csr.indptr[i + 1]] += \
                self.sample_mean_[i]
        X_csr.data += self.feature_mean_.take(X_csr.indices, mode='clip')

        return X_csr[samples, features].A[0]


class DLRecommender(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=None, n_components=10,
                 alpha=1., l1_ratio=1, algorithm='ridge',
                 n_epochs=1, batch_size=10,
                 memory=Memory(cachedir=None),
                 debug_folder=None,
                 ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.algorithm = algorithm
        self.n_components = n_components
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.memory = memory

        self.debug_folder = debug_folder

    def score(self, X, y, sample_weight=None):
        y_hat = self.predict(X)
        return - sqrt(mean_squared_error(y, y_hat, sample_weight=sample_weight))

    def predict(self, X):
        y_hat = np.zeros(X.shape[0])
        X_csr, (samples, features) = fm_format_to_csr_matrix(X, y_hat,
                                                             return_lists=True)
        for i in range(X_csr.shape[0]):
            X_csr.data[X_csr.indptr[i]: X_csr.indptr[i + 1]] += \
                self.sample_mean_[i]
            indices = X_csr.indices[X_csr.indptr[i]:X_csr.indptr[i + 1]]
            X_csr.data[X_csr.indptr[i]:
            X_csr.indptr[i + 1]] += self.code_[i].dot(
                self.dictionary_[:, indices])
        X_csr.data += self.feature_mean_.take(X_csr.indices, mode='clip')

        return X_csr[samples, features].A[0]

    def fit(self, X, y, **dump_kwargs):
        if self.debug_folder is not None:
            self.dump_init()
        X_ref = fm_format_to_csr_matrix(X, y)
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
            y_hat = self.predict(X)
            probe_score[i + 1] = sqrt(mean_squared_error(y_hat, y))
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
        if len(probe_score > 1):
            results['test_score'] = probe_score[1]
        if len(probe_score > 2):
            results['train_score'] = probe_score[2]
        with open(join(self.debug_folder, 'results.json'), 'w+') as f:
            json.dump(results, f)

        # np.save(join(self.debug_folder, 'dictionary'), self.dictionary_)
        # np.save(join(self.debug_folder, 'code'), self.code_)

        if debug_dict is not None:
            residuals = debug_dict['residuals']
            density = debug_dict['density']
            values = debug_dict['values']
            np.save(join(self.debug_folder, 'residuals'), residuals)
            np.save(join(self.debug_folder, 'density'), density)
            np.save(join(self.debug_folder, 'values'), values)
        draw_stats(self.debug_folder)


def fm_format_to_csr_matrix(X, y, return_lists=False):
    assert_array_equal(X.indptr, np.arange(0, (X.shape[0] + 1) * 2, 2))
    features = np.maximum(X.indices[1::2], X.indices[::2])
    samples = np.minimum(X.indices[1::2], X.indices[::2])
    assert_greater(features.min(), samples.max())
    n_samples = samples.max() + 1
    n_features = features.max() - features.min() + 1
    features -= features.min()

    X_csr = csr_matrix((y, (samples, features)), shape=(n_samples, n_features))
    if not return_lists:
        return X_csr
    else:
        return X_csr, (samples, features)


def main():
    output_dir = expanduser(join('~/output/dl_recommender/',
                                 datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                                  '-%M-%S')))
    os.makedirs(output_dir)

    random_state = check_random_state(0)
    mem = Memory(cachedir=expanduser("~/cache"), verbose=10)
    data = mem.cache(fetch_ml_10m)(expanduser('~/data/own/ml-10M100K'),
                                   remove_empty=True)
    permutation = random_state.permutation(data.shape[0])
    data = data[permutation]

    X, y = array_to_fm_format(data)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.1,
                                                        random_state=random_state)


    base_estimator = BaseRecommender()

    base_estimator.fit(X_train, y_train)
    score = base_estimator.score(X_test, y_test)
    print('RMSE base: %.3f' % score)

    dl_rec = DLRecommender(n_components=50,
                           batch_size=10,
                           n_epochs=1,
                           alpha=100,
                           memory=mem,
                           l1_ratio=0.,
                           debug_folder=None,
                           random_state=random_state)
    # dl_rec.fit(X_train, y_train)
    # score = dl_rec.score(X_test, y_test)
    # print('RMSE (non cv): %.3f' % score)

    dl_cv = GridSearchCV(dl_rec,
                          param_grid={'alpha': np.logspace(-3, 3, 10)},
                          n_jobs=16,
                          cv=ShuffleSplit(X_train.shape[0],
                                          n_iter=3, test_size=.1))
    dl_cv.fit(X_train, y_train)
    score = dl_cv.score(X_test, y_test)
    print('RMSE: %.3f' % score)
    print(dl_cv.grid_scores_)

if __name__ == '__main__':
    main()
