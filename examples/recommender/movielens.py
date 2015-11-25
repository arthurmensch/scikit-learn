import os
from os.path import expanduser, join
import functools

from sklearn.externals.joblib import Memory, delayed, Parallel
from sklearn.base import clone
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition.sparse_pca import IncrementalSparsePCA
from sklearn.utils import check_random_state, gen_batches
import matplotlib.pyplot as plt

def csr_rmse(y, y_pred):
    if np.isnan(y_pred.data).any():
        raise ValueError
    return np.sqrt(np.sum((y.data -
                           y_pred.data) ** 2) / y.nnz)


def draw_stats(debug_folder):
    residuals = np.load(join(debug_folder, 'residuals.npy'))
    density = np.load(join(debug_folder, 'density.npy'))
    values = np.load(join(debug_folder, 'values.npy'))
    dictionary = np.load(join(debug_folder, 'dictionary.npy'))
    code = np.load(join(debug_folder, 'code.npy'))
    # probe_score = np.load(join(debug_folder, 'probe_score.npy'))

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

    # fig = plt.figure()
    # plt.plot(np.arange(len(probe_score)), probe_score)
    # plt.savefig(join(debug_folder, 'probe_score.pdf'))
    # plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    plt.matshow(dictionary[:1000])
    plt.colorbar()
    plt.savefig(join(debug_folder, 'dictionary.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    plt.matshow(code[:1000])
    plt.colorbar()
    plt.savefig(join(debug_folder, 'code.pdf'))
    plt.close(fig)

def _fit_spca_recommender(X, incr_spca, seed=None, probe=None, probe_freq=100,
                          n_epochs=1):
    incr_spca = clone(incr_spca)
    this_random_state = check_random_state(seed)
    dict_init = (this_random_state.binomial(1, 0.5,
                                            size=(incr_spca.n_components,
                                                  X.shape[1])
                                            ) - .5) * 2
    incr_spca.set_params(dict_init=dict_init, random_state=seed)
    print("Learning dictionary")
    last_seen = 0
    for i in range(n_epochs):
        batches = gen_batches(X.shape[0], probe_freq)
        for batch in batches:
            last_seen = max(batch.stop, last_seen)
            incr_spca.partial_fit(X[batch], deprecated=False)
            dictionary = incr_spca.components_
            if np.isnan(dictionary).any():
                raise ValueError
            print("Done learning dictionary")
            print("Learning code")
            code = incr_spca.transform(X[:last_seen])
            if np.isnan(code).any():
                raise ValueError
            print("Done learning code")
            residuals = incr_spca.debug_info_['residuals']
            density = incr_spca.debug_info_['density']
            values = incr_spca.debug_info_['values']
            print("Computing probe score")
            yield dictionary, code, np.array(residuals)[:, np.newaxis], density, values,\
                  last_seen

def _fit_spca_recommender_(X, incr_spca, seed=None):
    incr_spca = clone(incr_spca)
    this_random_state = check_random_state(seed)
    dict_init = (this_random_state.binomial(1, 0.5,
                                            size=(incr_spca.n_components,
                                                  X.shape[1])
                                            ) - .5) * 2
    incr_spca.set_params(dict_init=dict_init, random_state=seed)
    print("Learning dictionary")
    incr_spca.fit(X)
    dictionary = incr_spca.components_
    print("Done learning dictionary")
    print("Learning code")
    code = incr_spca.transform(X)
    print("Done learning code")
    residuals = incr_spca.debug_info_['residuals']
    density = incr_spca.debug_info_['density']
    values = incr_spca.debug_info_['values']
    return dictionary, code,\
           np.array(residuals)[:, np.newaxis],\
           values,\
           np.array(density)[:, np.newaxis]


class SPCARecommender(BaseEstimator):
    def __init__(self, random_state=None, n_components=10, n_runs=1,
                 alpha=1, debug_folder=None, n_epochs=1,
                 n_jobs=1,
                 batch_size=10,
                 memory=Memory(cachedir=None)):
        self.alpha = alpha
        self.n_components = n_components
        self.n_runs = n_runs
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.memory = memory
        self.n_jobs = n_jobs
        self.debug_folder = debug_folder

    def fit(self, X, y=None, probe=None):
        X = X.copy()
        print("Centering data")
        _, self.global_mean_, \
        self.user_mean_, \
        self.movie_mean = csr_inplace_center_data(X)
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(0, np.iinfo(np.uint32).max,
                                     size=[self.n_runs])
        n_iter = X.shape[0] * self.n_epochs // self.batch_size
        self._incr_spca = IncrementalSparsePCA(n_components=self.n_components,
                                         alpha=self.alpha,
                                         l1_ratio=1,
                                         batch_size=self.batch_size,
                                         n_iter=n_iter,
                                         missing_values=0,
                                         verbose=10,
                                         transform_alpha=self.alpha,
                                         debug_info=True)
        # self._incr_spca = MiniBatchDictionaryLearning(
        #     n_components=self.n_components, alpha=self.alpha,
        #     batch_size=batch_size, fit_algorithm='cd',
        #     transform_algorithm='lasso_cd', n_iter=n_iter,
        #     missing_values=0, verbose=10, transform_alpha=self.alpha,
        #     debug_info=True)
        #
        if probe is None:
            res = Parallel(n_jobs=self.n_jobs, verbose=10)(delayed(self.memory.
            cache(_fit_spca_recommender_))(
            X, self._incr_spca, seed=seed) for seed in seeds)
            self.dictionary_, this_code, these_residuals, this_density, self.values_ = zip(*res)
            self.code_ = np.concatenate(this_code, axis=1)
            self.dictionary_ = np.concatenate(self.dictionary_, axis=0)
            self.residuals_ = np.concatenate(these_residuals, axis=1)
            self.density_ = np.concatenate(this_density, axis=1)
            self.values_ = self.values_[0]
            if self.debug_folder is not None:
                np.save(join(self.debug_folder, 'code'),
                        self.code_)
                np.save(join(self.debug_folder, 'dictionary'),
                        self.dictionary_)
                np.save(join(self.debug_folder, 'residuals'),
                        self.residuals_)
                np.save(join(self.debug_folder, 'density'),
                        self.density_)
                np.save(join(self.debug_folder, 'values'),
                            self.values_)
                draw_stats(self.debug_folder)
        else:
            self.probe_score_ = []
            print(self.n_epochs)
            for dictionary, code, residuals, density, values, last_seen\
                    in _fit_spca_recommender(
                    X, self._incr_spca, seed=seeds[0], n_epochs=self.n_epochs,
                    probe=probe, probe_freq=20):
                self.dictionary_ = dictionary
                self.code_ = code
                self.residuals_ = residuals
                self.density_ = density
                self.values_ = values
                probe_score = []
                print(np.abs(self.dictionary_).max())
                print(np.abs(self.code_).max())
                for this_probe in probe:
                    probe_score.append(self.score(this_probe[:last_seen]))
                self.probe_score_.append(probe_score)
                for score in self.probe_score_[-1]:
                    print('RMSE: %.3f' % score)
                if self.debug_folder is not None:
                    np.save(join(self.debug_folder, 'code'),
                            self.code_)
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
                    draw_stats(self.debug_folder)


    def transform(self, X):
        # Use only X sparsity structure
        X = X.copy()
        X.data[:] = self.global_mean_
        for i in range(X.shape[0]):
            if X.indptr[i] < X.indptr[i + 1]:
                indices = X.indices[X.indptr[i]:X.indptr[i + 1]]
                X.data[X.indptr[i]:X.indptr[i + 1]] += self.user_mean_[i]
        X.data += self.movie_mean.take(X.indices, mode='clip')

        infered_rel = self.code_[i].dot(self.dictionary_[:, indices])
        if np.isnan(infered_rel.data).any():
            raise ValueError
        # infered_rel -= infered_rel.mean()
        X.data[X.indptr[i]: X.indptr[i + 1]] += infered_rel

        X.data[X.indptr[i]: X.indptr[i + 1]] = np.minimum(X.data[X.indptr[i]: X.indptr[i + 1]], 5)
        X.data[X.indptr[i]: X.indptr[i + 1]] = np.maximum(X.data[X.indptr[i]: X.indptr[i + 1]], 1)
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


class SPCARecommenderCV(BaseEstimator):
    def __init__(self, random_state=None, n_components=10, n_runs=1,
                 alphas=[1], debug_folder=None, batch_size=10,
                 n_epochs=1, n_jobs=1,
                 memory=Memory(cachedir=None)):
        self.alphas = alphas
        self.n_components = n_components
        self.n_runs = n_runs
        self.random_state = random_state
        self.debug_folder = debug_folder
        self.n_epochs = n_epochs
        self.n_jobs = n_jobs
        self.memory = memory

    def fit(self, X, y=None):
        scorer = functools.partial(recommender_scorer,
                                   n_splits=1, test_size=0.2,
                                   random_state=0)
        recommender = SPCARecommender(
            random_state=self.random_state,
            n_components=self.n_components,
            n_runs=self.n_runs,
            n_epochs=self.n_epochs,
            n_jobs=1,
            memory=self.memory
        )
        scores = Parallel(n_jobs=self.n_jobs,
                          verbose=10)(delayed(grid_point_fit)(recommender,
                                                              X,
                                                              scorer,
                                                              alpha,
                                                              self.debug_folder)
                                      for alpha
                                      in self.alphas)
        scores = np.array(scores)
        self.alpha_ = self.alphas[np.argmax(scores)]
        recommender.set_params(alpha=self.alpha_)
        self.recommender_ = recommender
        self.recommender_.fit(X)

    def transform(self, X, y=None):
        return self.recommender_.transform(self, X)

    def score(self, X):
        return self.recommender_.score(X)


def recommender_scorer(estimator, X, y=None, n_splits=1, test_size=0.2,
                       random_state=0):
    splits = CsrRowStratifiedShuffleSplit(X, test_size=test_size,
                                          n_splits=n_splits,
                                          random_state=random_state)
    score = 0
    for X_train, X_test in splits:
        estimator.fit(X_train)
        score += estimator.score(X_test)
    return score / n_splits


def fetch_dataset(datafile='/home/arthur/data/own/ml-20m/ratings.csv',
                  n_users=None, n_movies=None):
    df = pd.read_csv(datafile)

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
    m_col = np.zeros(X.shape[1])
    m_row = np.zeros(X.shape[0])
    for i in range(3):
        row_nnz = X.getnnz(axis=1)
        col_nnz = X.getnnz(axis=0)
        row_nnz[row_nnz == 0] = 1
        col_nnz[col_nnz == 0] = 1
        m_col = X.sum(axis=1).A[:, 0] / row_nnz - m_row.mean()
        m_row = X.sum(axis=0).A[0, :] / col_nnz - m_col.mean()
        m_global = m_col.mean() + m_row.mean()
        m_row -= m_row.mean()
        m_col -= m_col.mean()
    for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
        if left < right:
            X.data[left:right] -= m_col[i]
    X.data -= m_row.take(X.indices, mode='clip')
    return X, m_global, m_col, m_row


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


def run():
    random_state = check_random_state(0)
    mem = Memory(cachedir=expanduser("~/cache"), verbose=10)
    print("Loading dataset")
    X = mem.cache(fetch_dataset)(expanduser('~/ml-20m/ratings.csv'))
    X = X[:10000]
    print("Done loading dataset")
    splits = list(CsrRowStratifiedShuffleSplit(X, test_size=0.1, n_splits=1,
                                               random_state=random_state))
    alphas = [.01, .1, 1, 10]
    for i, (X_train, X_test) in enumerate(splits):

        # recommender = SPCARecommender(n_components=50,
        #                               batch_size=1,
        #                               n_epochs=3,
        #                               n_runs=1,
        #                               random_state=random_state,
        #                               alpha=.1,
        #                               memory=mem,
        #                               debug_folder=
        #                               expanduser(
        #                                   '~/test_recommender_output'))
        recommender = SPCARecommenderCV(n_components=50,
                                        n_epochs=5,
                                        n_runs=1,
                                        n_jobs=4,
                                        random_state=random_state,
                                        alphas=alphas,
                                        batch_size=1,
                                        memory=mem,
                                        debug_folder=expanduser('~/test_recommender_output'))
        recommender.fit(X_train)
        score = recommender.score(X_test)
        print("RMSE: %.2f" % score)


if __name__ == '__main__':
    run()
