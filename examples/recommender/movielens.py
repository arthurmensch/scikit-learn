from os.path import expanduser

from joblib import Memory
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition.sparse_pca import IncrementalSparsePCA
from sklearn.utils import check_random_state

import matplotlib.pyplot as plt

def fetch_dataset(datafile='/home/arthur/data/own/ml-20m/ratings.csv'):
    df = pd.read_csv(datafile)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    df.reset_index(drop=True, inplace=True)
    df.set_index(['userId', 'movieId'], inplace=True)
    df.sort_index(level=['userId', 'movieId'], inplace=True)

    ratings = df['rating']

    n_users = ratings.index.get_level_values(0).max()
    n_movies = ratings.index.get_level_values(1).max()

    X = csr_matrix((ratings, (ratings.index.get_level_values(0) - 1,
                             ratings.index.get_level_values(1) -1)),
                   shape=(n_users, n_movies))
    return X[:, :10000]


def split_dataset(X):
    # FIXME use sklearn
    X_test = X.copy()
    X_train = X.copy()

    for i in range(X.shape[0]):
        _, non_zero_y = X[i].nonzero()
        X_test[i, non_zero_y[::2]] = 0
        X_train[i, non_zero_y[1::2]] = 0

    X_test.eliminate_zeros()
    X_train.eliminate_zeros()

    return X_train, X_test


def center_non_zero_data_along_row(X):
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
    mem = Memory(cachedir=expanduser("~/cache"))
    X = mem.cache(fetch_dataset)(datafile='/home/arthur/data/own/ml-20m/ratings.csv')
    X_test, X_train = mem.cache(split_dataset)(X)
    X_train, mean_train = mem.cache(center_non_zero_data_along_row)(X_train[
                                                                    :1000])
    random_state = check_random_state(0)
    dict_init = (random_state.binomial(1, 0.5, size=(40, X.shape[1])) - .5) * 2
    sparse_pca = IncrementalSparsePCA(n_components=40, dict_init=dict_init,
                                      alpha=0.01, batch_size=1,
                                      n_iter=10000, missing_values=0,
                                      verbose=10, transform_alpha=1,
                                      random_state=random_state,
                                      debug_info=True)
    sparse_pca.fit(X_train)
    code = sparse_pca.transform(X_train[:1000])
    np.save('code', code)
    np.save('components', sparse_pca.components_)

    residuals = sparse_pca.debug_info_['residuals']

    plt.figure(figsize=(4.2, 4))
    plt.plot(np.arange(len(residuals)), residuals, label='Residuals')
    plt.plot(np.arange(len(residuals)), sparse_pca.debug_info_['norm_cost'], label='Norm cost')
    plt.plot(np.arange(len(residuals)), sparse_pca.debug_info_['objective_cost'], label='Objective cost')
    plt.plot(np.arange(len(residuals)), sparse_pca.debug_info_['penalty_cost'], label='Penalty cost')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run()