import copy
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


def csr_rmse(y, y_pred):
    return np.sqrt(np.sum((y.data -
                           y_pred.data) ** 2) / y.nnz)


def _fit_spca_recommender(X, incr_spca, seed=None):
    incr_spca = clone(incr_spca)
    this_random_state = check_random_state(seed)
    dict_init = (this_random_state.binomial(1, 0.5,
                                            size=(incr_spca.n_components,
                                                  X.shape[1])
                                            ) - .5) * 2
    incr_spca.set_params(dict_init=dict_init, random_state=seed)
    print("Learning dictionary")
    incr_spca.partial_fit(X)
    dictionary = incr_spca.components_
    print("Done learning dictionary")
    print("Learning code")
    code = incr_spca.transform(X)
    print("Done learning code")
    residuals = incr_spca.debug_info_['residuals']
    return dictionary, code, np.array(residuals)[:, np.newaxis]


class SPCARecommender(BaseEstimator):
    def __init__(self, random_state=None, n_components=10, n_runs=1,
                 alpha=1, debug_folder=None, n_epochs=1,
                 n_jobs=1,
                 memory=Memory(cachedir=None)):
        self.alpha = alpha
        self.n_components = n_components
        self.n_runs = n_runs
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.memory = memory
        self.n_jobs = n_jobs
        self.debug_folder = debug_folder

    def _pre_fit(self, X):
        print("Centering data")
        _, self.user_mean_ = csr_inplace_row_center_data(X)

    def fit(self, X, y=None):
        if not hasattr(self, 'user_mean_'):
            self._pre_fit(X)
        self.partial_fit(X, y=None)

    def partial_fit(self, X, y=None, centered=False):
        if not centered:
            X = X.copy()
            X, _ = csr_inplace_row_center_data(X)
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(0, np.iinfo(np.uint32).max,
                                     size=[self.n_runs])
        batch_size = 20
        n_iter = X.shape[0] * self.n_epochs // batch_size
        print(n_iter)
        if not hasattr(self, '_incr_spca'):
            self._incr_spca = IncrementalSparsePCA(n_components=self.n_components,
                                             alpha=self.alpha,
                                             batch_size=batch_size,
                                             n_iter=n_iter,
                                             missing_values=0,
                                             verbose=10,
                                             transform_alpha=self.alpha,
                                             debug_info=True)
        res = Parallel(n_jobs=self.n_jobs, verbose=10)(delayed(self.memory.
            cache(_fit_spca_recommender))(
            X, self._incr_spca, seed=seed) for seed in seeds)
        self.dictionary_, this_code, these_residuals = zip(*res)
        this_code = np.concatenate(this_code, axis=1)
        self.dictionary_ = np.concatenate(self.dictionary_, axis=0)
        these_residuals = np.concatenate(these_residuals, axis=1)
        if hasattr(self, 'residuals_'):
            self.code_ = this_code
            # np.concatenate((self.code_, this_code), axis=0)
            self.residuals_ = np.concatenate((self.residuals_,
                                              these_residuals), axis=0)
        else:
            self.code_ = this_code
            self.residuals_ = these_residuals
        if self.debug_folder is not None:
            np.save(join(self.debug_folder, 'code'),
                    self.code_)
            np.save(join(self.debug_folder, 'dictionary'),
                    self.dictionary_)
            np.save(join(self.debug_folder, 'residual'),
                    self.residuals_)

    def transform(self, X, y=None):
        # Use only X sparsity structure
        X = X.copy()
        X.data[:] = 0
        print(X.shape[0])
        for i in range(X.shape[0]):
            if X.indptr[i] != X.indptr[i + 1]:
                indices = X.indices[X.indptr[i]:X.indptr[i + 1]]
                X.data[X.indptr[i]:X.indptr[i + 1]] = self.user_mean_[i]
                X.data[X.indptr[i]:
                X.indptr[i + 1]] += self.code_[i].dot(
                    self.dictionary_[:, indices])
        return X

    def score(self, X):
        X_pred = self.transform(X)
        if self.debug_folder is not None:
            np.save(join(self.debug_folder, 'X_pred'),
                    X_pred.data)
            np.save(join(self.debug_folder, 'X'),
                    X.data)
        return csr_rmse(X, X_pred)


def grid_point_fit(recommender, X, scorer, alpha):
    this_recommender = clone(recommender)
    this_recommender.memory = recommender.memory
    recommender.set_params(alpha=alpha)
    return scorer(recommender, X)


class SPCARecommenderCV(BaseEstimator):
    def __init__(self, random_state=None, n_components=10, n_runs=1,
                 alphas=[1], debug_folder=None, n_epochs=1, n_jobs=1,
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
            debug_folder=self.debug_folder,
            n_epochs=self.n_epochs,
            n_jobs=1,
            memory=self.memory
        )
        scores = Parallel(n_jobs=self.n_jobs,
                          verbose=10)(delayed(grid_point_fit)(recommender,
                                                              X,
                                                              scorer,
                                                              alpha) for alpha
                                      in self.alphas)
        scores = np.array(scores)
        self.alpha_ = self.alphas[np.argmax(scores)]
        recommender.set_params(alpha=self.alpha_)
        self.recommender_ = recommender
        self.recommender_.fit(X)

    def transform(self, X, y=None):
        self.recommender_.transform(self, X)

    def score(self, X):
        self.recommender_.score(self, X)


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


def csr_inplace_row_center_data(X):
    """
    X: csc sparse matrix
    """
    X_mean = np.zeros(X.shape[0])
    # User centering
    for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
        X_mean[i] = X.data[left:right].mean()
        X.data[left:right] -= X_mean[i]

    return X, X_mean


def run():
    random_state = check_random_state(0)
    mem = Memory(cachedir=expanduser("~/cache"), verbose=10)
    print("Loading dataset")
    X = mem.cache(fetch_dataset)(
        datafile='/home/arthur/data/own/ml-20m/ratings.csv')
    print("Done loading dataset")
    splits = list(CsrRowStratifiedShuffleSplit(X, test_size=0.1, n_splits=1,
                                               random_state=random_state))
    alphas = np.array([1, 10, 100])
    batches = gen_batches(X.shape[0], 500)
    for i, (X_train, X_test) in enumerate(splits):
        # if i == 0:
        # recommender = SPCARecommenderCV(n_components=50,
        #                                 n_epochs=1,
        #                                 n_runs=1,
        #                                 n_jobs=3,
        #                                 random_state=random_state,
        #                                 alphas=alphas,
        #                                 memory=mem)
        # alpha = recommender.alpha_
        # else:
        recommender = SPCARecommender(n_components=20,
                                      n_epochs=1,
                                      n_runs=1,
                                      random_state=random_state,
                                      alpha=10,
                                      memory=mem,
                                      debug_folder=
                                      expanduser(
                                          '~/test_recommender_output'))
        recommender._pre_fit(X_train)
        for batch in batches:
            recommender.partial_fit(X_train[batch], centered=True)
            score = recommender.score(X_test[:batch.stop])
            print("RMSE: %.2f" % score)


if __name__ == '__main__':
    run()
