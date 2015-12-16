from os.path import join

import pandas as pd
from scipy.sparse import csr_matrix


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
        rated_movies = (X.getnnz(axis=0) > 0)
        X = X[:, rated_movies]
        rating_users = (X.getnnz(axis=1) > 0)
        X = X[rating_users, :]
    return X