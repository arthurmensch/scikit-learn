import datetime
import fnmatch
import os
from os.path import expanduser, join
import functools
import json
from math import sqrt
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.externals.joblib import Memory, delayed, Parallel
from sklearn.base import clone
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition.sparse_pca import IncrementalSparsePCA
from sklearn.linear_model import ridge_regression
from sklearn.utils import check_random_state, gen_batches
import matplotlib.pyplot as plt


def csr_rmse(y, y_pred):
    if np.isnan(y_pred.data).any():
        raise ValueError
    return sqrt(np.sum((y.data -
                        y_pred.data) ** 2) / y.nnz)


def draw_stats(debug_folder):
    residuals = np.load(join(debug_folder, 'residuals.npy'))
    density = np.load(join(debug_folder, 'density.npy'))
    values = np.load(join(debug_folder, 'values.npy'))
    dictionary = np.load(join(debug_folder, 'dictionary.npy'))
    code = np.load(join(debug_folder, 'code.npy'))
    probe_score = np.load(join(debug_folder, 'probe_score.npy'))
    count_seen_features = np.load(join(debug_folder,
                                       'count_seen_features.npy'))

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
    plt.plot(np.arange(len(probe_score)), probe_score)
    plt.savefig(join(debug_folder, 'probe_score.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    plt.matshow(dictionary[:, :1000])
    plt.colorbar()
    plt.savefig(join(debug_folder, 'dictionary.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    plt.matshow(code[:1000].reshape((-1, code.shape[1] * 2)))
    plt.colorbar()
    plt.savefig(join(debug_folder, 'code.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    plt.plot(np.arange(len(count_seen_features)), count_seen_features)
    plt.savefig(join(debug_folder, 'count_seen_features.pdf'))
    plt.close(fig)

    plt.close('all')


# def _fit_spca_recommender(X, incr_spca, seed=None, probe=None, probe_freq=100,
#                           n_epochs=1):
#     incr_spca = clone(incr_spca)
#     this_random_state = check_random_state(seed)
#     # dict_init = (this_random_state.binomial(1, 0.5,
#     #                                         size=(incr_spca.n_components,
#     #                                               X.shape[1])
#     #                                         ) - .5) * 2
#     dict_init = this_random_state.randn(incr_spca.n_components,
#                                         X.shape[1])
#     incr_spca.set_params(dict_init=dict_init, random_state=seed)
#     print("Learning dictionary")
#     last_seen = 0
#     for i in range(n_epochs):
#         print(i)
#         batches = gen_batches(X.shape[0], probe_freq * incr_spca.batch_size)
#         for batch in batches:
#             last_seen = max(batch.stop, last_seen)
#             incr_spca.partial_fit(X[batch], deprecated=False)
#             dictionary = incr_spca.components_
#             if np.isnan(dictionary).any():
#                 raise ValueError
#             print("Done learning dictionary")
#             print("Learning code")
#             code = incr_spca.transform(X[:last_seen])
#             if np.isnan(code).any():
#                 raise ValueError
#             print("Done learning code")
#             residuals = incr_spca.debug_info_['residuals']
#             density = incr_spca.debug_info_['density']
#             values = incr_spca.debug_info_['values']
#             count_seen_features = incr_spca.debug_info_['count_seen_features']
#             print("Computing probe score")
#             yield dictionary, code, np.array(residuals)[:,
#                                     np.newaxis], density, values, \
#                   count_seen_features, last_seen
#
#
# def _fit_spca_recommender_(X, incr_spca, seed=None):
#     incr_spca = clone(incr_spca)
#     this_random_state = check_random_state(seed)
#     # dict_init = (this_random_state.binomial(1, 0.5,
#     #                                         size=(incr_spca.n_components,
#     #                                               X.shape[1])
#     #                                         ) - .5) * 2
#     dict_init = this_random_state.randn(incr_spca.n_components,
#                                         X.shape[1])
#     incr_spca.set_params(dict_init=dict_init, random_state=seed)
#     print("Learning dictionary")
#     incr_spca.fit(X)
#     dictionary = incr_spca.components_
#     print("Done learning dictionary")
#     print("Learning code")
#     code = incr_spca.transform(X)
#     print("Done learning code")
#     residuals = incr_spca.debug_info_['residuals']
#     density = incr_spca.debug_info_['density']
#     values = incr_spca.debug_info_['values']
#     return dictionary, code, \
#            np.array(residuals)[:, np.newaxis], \
#            values, \
#            np.array(density)[:, np.newaxis]


class BaseRecommender(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X):
        X = X.copy()
        print("Centering data")
        _, self.global_mean_, \
        self.user_mean_, \
        self.movie_mean_ = csr_inplace_center_data(X)

    def transform(self, X):
        X = X.copy()
        X.data[:] = self.global_mean_
        for i in range(X.shape[0]):
            if X.indptr[i] < X.indptr[i + 1]:
                X.data[X.indptr[i]:X.indptr[i + 1]] += self.user_mean_[i]
        X.data += self.movie_mean_.take(X.indices, mode='clip')
        return X

    def score(self, X):
        X_pred = self.transform(X)
        return csr_rmse(X, X_pred)


class ALSRecommender(BaseRecommender):
    def __init__(self, alpha_dict=1, alpha_code=1, n_components=10,
                 n_iter=10):
        alpha_dict = alpha_dict
        alpha_code = alpha_code
        n_components = n_components
        n_iter = n_iter

    def fit(self, X, y=None):
        BaseRecommender.fit(self, X)

        dictionary = np.zeros((X.shape[0], self.n_components))
        code = np.zeros((self.n_components, X.shape[1]))

        for i in range(self.n_iter):
            for j in range(X.shape[0]):
                data = X.data[X.indptr[i]:X.indptr[i]]
                dictionary[j] = ridge_regression(code, data,
                                                 alpha=self.alpha_code)
                dictionary[j] = ridge_regression(code, data,
                                                 alpha=self.alpha_code)


class SPCARecommender(BaseEstimator):
    def __init__(self, random_state=None, n_components=10, n_runs=1,
                 alpha=1., l1_ratio=1, debug_folder=None, n_epochs=1,
                 batch_size=10,
                 dict_penalty=0,
                 memory=Memory(cachedir=None)):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.dict_penalty = dict_penalty
        self.n_components = n_components
        self.n_runs = n_runs
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.memory = memory
        self.debug_folder = debug_folder

    def fit(self, X, y=None, probe=None, probe_freq=100):
        X = X.copy()
        print("Centering data")
        _, self.global_mean_, \
        self.user_mean_, \
        self.movie_mean_ = csr_inplace_center_data(X)

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(0, np.iinfo(np.uint32).max,
                                     size=[self.n_runs])
        n_iter = X.shape[0] * self.n_epochs // self.batch_size
        incr_spca = IncrementalSparsePCA(n_components=self.n_components,
                                               alpha=self.alpha,
                                               l1_ratio=self.l1_ratio,
                                               batch_size=self.batch_size,
                                               n_iter=n_iter,
                                               missing_values=0,
                                               verbose=10,
                                               transform_alpha=self.alpha,
                                               debug_info=True)
        self.probe_score_ = []
        self.code_ = np.zeros((X.shape[0], self.n_components))
        last_seen = 0
        for i in range(self.n_epochs):
            batches = gen_batches(X.shape[0],
                                  probe_freq * incr_spca.batch_size)
            print(i)
            for batch in batches:
                last_seen = max(batch.stop, last_seen)
                incr_spca.partial_fit(X[batch], deprecated=False)

                self.code_[:last_seen] = incr_spca.transform(X[:last_seen])
                self.dictionary_ = incr_spca.components_
                self.residuals_ = incr_spca.debug_info_['residuals']
                self.density_ = incr_spca.debug_info_['density']
                self.values_ = incr_spca.debug_info_['values']
                self.count_seen_features_ = incr_spca.debug_info_[
                    'count_seen_features']
                if probe is not None:
                    probe_score = []
                    for this_probe in probe:
                            probe_score.append(self.score(this_probe))
                    self.probe_score_.append(probe_score)
                    for score in self.probe_score_[-1]:
                        print('RMSE: %.3f' % score)
                    if self.debug_folder is not None:
                        np.save(join(self.debug_folder, 'code'),
                                self.code_)
                        with open(join(self.debug_folder, 'results.json'),
                                  'r') as f:
                            results = json.load(f)
                        results['test_score'] = probe_score[0]
                        results['train_score'] = probe_score[1]
                        with open(join(self.debug_folder, 'results.json'),
                                  'w+') as f:
                            json.dump(results, f)
                        np.save(join(self.debug_folder, 'dictionary'),
                                self.dictionary_)
                        np.save(join(self.debug_folder, 'residuals'),
                                self.residuals_)
                        np.save(join(self.debug_folder, 'probe_score'),
                                np.array(self.probe_score_))
                        np.save(join(self.debug_folder, 'density'),
                                self.density_)
                        np.save(join(self.debug_folder, 'values'),
                                self.values_)
                        np.save(join(self.debug_folder, 'count_seen_features'),
                                self.count_seen_features_)
                        draw_stats(self.debug_folder)

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
        # inter_mean /= X.shape[0]
        # inter_mean = sqrt(inter_mean)
        # print(inter_mean)
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


def grid_point_fit(recommender, X, scorer, alpha, root_debug):
    this_recommender = clone(recommender)
    this_recommender.memory = recommender.memory
    try:
        os.makedirs(join(root_debug, str(alpha).replace('.', '_')))
    except:
        pass
    recommender.set_params(alpha=alpha,
                           debug_folder=join(root_debug,
                                             str(alpha).replace('.', '_')))
    return scorer(recommender, X)


# class SPCARecommenderCV(BaseEstimator):
#     def __init__(self, random_state=None, n_components=10, n_runs=1,
#                  alphas=[1], debug_folder=None, batch_size=10,
#                  n_epochs=1, n_jobs=1,
#                  memory=Memory(cachedir=None)):
#         self.alphas = alphas
#         self.n_components = n_components
#         self.n_runs = n_runs
#         self.random_state = random_state
#         self.debug_folder = debug_folder
#         self.n_epochs = n_epochs
#         self.n_jobs = n_jobs
#         self.memory = memory
#         self.batch_size = batch_size
#
#     def fit(self, X, y=None):
#         scorer = functools.partial(recommender_scorer,
#                                    n_splits=1, test_size=0.2,
#                                    random_state=0)
#         recommender = SPCARecommender(
#             random_state=self.random_state,
#             n_components=self.n_components,
#             n_runs=self.n_runs,
#             n_epochs=self.n_epochs,
#             batch_size=self.batch_size,
#             n_jobs=1,
#             memory=self.memory
#         )
#         scores = Parallel(n_jobs=self.n_jobs,
#                           verbose=10)(delayed(grid_point_fit)(recommender,
#                                                               X,
#                                                               scorer,
#                                                               alpha,
#                                                               self.debug_folder)
#                                       for alpha
#                                       in self.alphas)
#         scores = np.array(scores)
#         self.alpha_ = self.alphas[np.argmax(scores)]
#         recommender.set_params(alpha=self.alpha_)
#         self.recommender_ = recommender
#         self.recommender_.fit(X)
#
#     def transform(self, X, y=None):
#         return self.recommender_.transform(self, X)
#
#     def score(self, X):
#         return self.recommender_.score(X)
#
#
# def recommender_scorer(estimator, X, y=None, n_splits=1, test_size=0.2,
#                        random_state=0):
#     splits = CsrRowStratifiedShuffleSplit(X, test_size=test_size,
#                                           n_splits=n_splits,
#                                           random_state=random_state)
#     score = 0
#     for X_train, X_test in splits:
#         estimator.fit(X_train)
#         score += estimator.score(X_test)
#     return score / n_splits



def fetch_dataset(datadir='/home/arthur/data/own/ml-20m',
                  n_users=None, n_movies=None):
    df = pd.read_csv(join(datadir, 'ratings.csv'))

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
    return X[:n_users, :n_movies]


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
        rated_movies = (X.getnnz(axis=0) != 0)
        X = X[:, rated_movies]
        print(np.where((X.getnnz(axis=0) == 0)))
        rating_users = (X.getnnz(axis=1) != 0)
        X = X[rating_users, :]
    return X


def CsrRowStratifiedShuffleSplit(X, n_splits=5, test_size=0.1, train_size=None,
                                 random_state=None):
    if train_size is None:
        train_size = 1 - test_size

    random_state = check_random_state(random_state)
    seeds = random_state.randint(0, np.iinfo(np.uint32).max, size=[n_splits])
    for seed in seeds:
        train_indptr = np.zeros(len(X.indptr))
        test_indptr = np.zeros(len(X.indptr))
        train_mask = []
        test_mask = []
        for i in range(X.shape[0]):
            this_random_state = check_random_state(seed)
            n_row_elem = X.indptr[i + 1] - X.indptr[i]
            permutation = this_random_state.permutation(n_row_elem)
            train_lim = int(train_size * n_row_elem)
            test_lim = int((1 - test_size) * n_row_elem)
            train_mask += (permutation[:train_lim] + X.indptr[i]).tolist()
            test_mask += (permutation[test_lim:] + X.indptr[i]).tolist()
            train_indptr[i + 1] = len(train_mask)
            test_indptr[i + 1] = len(test_mask)
        X_train = csr_matrix((X.data[train_mask],
                              X.indices[train_mask], train_indptr))
        X_test = csr_matrix((X.data[test_mask],
                             X.indices[test_mask], test_indptr))
        yield X_train, X_test


def csr_inplace_center_data(X):
    w_global = 0

    acc_u = np.zeros(X.shape[0])
    acc_m = np.zeros(X.shape[1])

    n_u = X.getnnz(axis=1)
    n_m = X.getnnz(axis=0)
    n_u[n_u == 0] = 1
    n_m[n_m == 0] = 1
    for i in range(10):
        # Y = X.copy()
        # Y.data[:] -= w_global
        # w_global = w_u.mean() + w_m.mean()
        w_u = X.sum(axis=1).A[:, 0] / n_u
        for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
            X.data[left:right] -= w_u[i]
        w_m = X.sum(axis=0).A[0] / n_m
        X.data -= w_m.take(X.indices, mode='clip')

        acc_u += w_u
        acc_m += w_m

    return X, w_global, acc_u, acc_m


def csr_inplace_row_center_data(X):
    """
    X: csc sparse matrix
    """
    X_mean = np.zeros(X.shape[0])
    # User centering
    for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
        if left < right:
            X_mean[i] = X.data[left:right].mean()
            X.data[left:right] -= X_mean[i]
        else:
            X_mean[i] = 0
    return X, X_mean


def csr_inplace_column_center_data(X):
    """
    X: csc sparse matrix
    """
    X_mean = np.zeros(X.shape[1])
    X.data -= X_mean.take(X.indices, mode='clip')
    return X, X_mean


def fit_and_dump(recommender, X_train, X_test):
    result_dict = {'n_components': recommender.n_components,
                   'l1_ratio': recommender.l1_ratio,
                   'alpha': recommender.alpha,
                   'batch_size': recommender.batch_size}
    with open(join(recommender.debug_folder, 'results.json'), 'w+') as f:
        json.dump(result_dict, f)
    recommender.fit(X_train, probe=[X_test, X_train], probe_freq=200)
    score = recommender.score(X_test)
    with open(join(recommender.debug_folder, 'results.json'), 'r') as f:
        result_dict = json.load(f)
    result_dict['final_score'] = score
    with open(join(recommender.debug_folder, 'results.json'), 'w+') as f:
        json.dump(result_dict, f)


def gather_results(output_dir):
    full_dict_list = []
    for dirpath, dirname, filenames in os.walk(output_dir):
        for filename in fnmatch.filter(filenames, 'results.json'):
            with open(join(dirpath, filename), 'r') as f:
                exp_dict = json.load(f)
                exp_dict['path'] = dirpath
                full_dict_list.append(exp_dict)
    results = pd.DataFrame(full_dict_list, columns=['path', 'n_components',
                                                    'l1_ratio',
                                                    'reduction_method',
                                                    'alpha',
                                                    'batch_size',
                                                    'test_score',
                                                    'train_score',
                                                    'final_score'])

    results.sort_values(by=['path',
                            'n_components', 'l1_ratio', 'reduction_method',
                            'alpha', 'batch_size',
                            'test_score', 'train_score',
                            'final_score'], inplace=True)
    results.to_csv(join(output_dir, 'results.csv'))


def run(n_jobs=1):
    random_state = check_random_state(0)
    mem = Memory(cachedir=expanduser("~/cache"), verbose=10)
    print("Loading dataset")
    X = mem.cache(fetch_ml_10m)(expanduser('~/data/own/ml-10M100K'),
                                remove_empty=True)
    X = X[random_state.permutation(X.shape[0])]
    print("Done loading dataset")
    splits = list(CsrRowStratifiedShuffleSplit(X, test_size=0.1, n_splits=1,
                                               random_state=random_state))
    X_train, X_test = splits[0]

    recommender = BaseRecommender()
    recommender.fit(X_train)
    score = recommender.score(X_test)
    print("Unbiasing RMSE: %.2f" % score)
    output_dir = expanduser(join('~/output/movielens/',
                                 datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                                  '-%M-%S')))
    os.makedirs(output_dir)
    recommenders = [SPCARecommender(n_components=n_components,
                                    batch_size=batch_size,
                                    n_epochs=6,
                                    n_runs=1,
                                    alpha=alpha,
                                    memory=mem,
                                    l1_ratio=l1_ratio,
                                    random_state=random_state)
                    for n_components in [50]
                    for batch_size in [1, 10, 100]
                    for l1_ratio in np.linspace(0, 1, 3)
                    for alpha in np.logspace(-2, 2, 5)]
    # recommenders = [SPCARecommender(n_components=20,
    #                                 batch_size=1,
    #                                 alpha=0.1,
    #                                 n_epochs=1,
    #                                 l1_ratio=1,
    #                                 random_state=random_state)]
    for i, recommender in enumerate(recommenders):
        path = join(output_dir, "experiment_%i" % i)
        recommender.set_params(debug_folder=join(path))
        os.makedirs(path)
    print(n_jobs)
    Parallel(n_jobs=n_jobs, verbose=10, max_nbytes=0)(
            delayed(fit_and_dump)(recommender, X_train, X_test)
        for recommender in recommenders)


if __name__ == '__main__':
    run(n_jobs=16)
    # gather_results(expanduser('~/output/movielens/2015-12-02_14-38-20'))
