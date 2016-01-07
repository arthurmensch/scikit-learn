import tarfile
from os.path import join

import numpy as np
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


def fetch_nf(datadir='/volatile/arthur/nf_prize'):
    probe_users, probe_movies = fetch_nf_probe(datadir)
    df = []
    with tarfile.open(join(datadir, 'training_set.tar')) as tar:
        for i, member in enumerate(tar.getmembers()[1:400]):
            if i % 100 == 0:
                print('%i movies loaded' % i)
            file = tar.extractfile(member)
            movieId = int(file.readline()[:-2])
            this_df = pd.read_csv(file,
                                  header=None,
                                  names=['userId', 'ratings', 'date'],
                                  parse_dates=[2])
            this_df['movieId'] = movieId
            df.append(this_df)
    df = pd.concat(df)
    df.reset_index(drop=True, inplace=True)
    df.set_index(['userId', 'movieId'], inplace=True)
    df.sort_index(level=['userId', 'movieId'], inplace=True)

    idx = np.r_([probe_users, probe_movies]).T
    df_test = df.loc[idx]

    X = csr_matrix((df['ratings'], (df['userId'] - 1, df['movieId'] - 1)),
                   shape=(df['userId'].max(), df['movieId'].max()))
    return X


def fetch_nf_probe(datadir='/volatile/arthur/nf_prize'):
    movieId = []
    userId = []
    with open(join(datadir, 'probe.txt'), 'r') as f:
        for line in f:
            if line.strip().endswith(':'):
                current = int(line[:-2])
            else:
                movieId.append(int(line))
                userId.append(current)
    return np.array(userId), np.array(movieId)
