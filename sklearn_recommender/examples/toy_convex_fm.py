import array

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn_recommender import ConvexFM


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

n_samples, n_features = 1000, 50
rank = 5
length = 5
X, y = make_multinomial_fm_dataset(n_samples, n_features, rank, length,
                                   random_state=0)
X, X_val, y, y_val = train_test_split(X, y, test_size=0.25, random_state=0)
y += 0.01 * np.random.RandomState(0).randn(*y.shape)

# try ridge
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=np.logspace(-4, 4, num=9, base=10),
                fit_intercept=False)
ridge.fit(X, y)
y_val_pred = ridge.predict(X_val)
print('RidgeCV validation RMSE={}'.format(
    np.sqrt(mean_squared_error(y_val, y_val_pred))))

# convex factorization machine path
fm = ConvexFM(fit_linear=True, warm_start=False, max_iter=20, tol=1e-4,
              max_iter_inner=50, fit_intercept=True,
              eigsh_kwargs={'tol': 0.1})

if True:
    for alpha in (0.01, 0.1, 1, 10):
        for beta in (10000, 500, 150, 100, 50, 1, 0.001):
            fm.set_params(alpha=alpha, beta=beta)
            fm.fit(X, y)
            y_val_pred = fm.predict(X_val)
            print("α={} β={}, rank={}, validation RMSE={:.2f}".format(
                alpha,
                beta,
                fm.rank_,
                np.sqrt(mean_squared_error(y_val, y_val_pred))))

if False:
    fm.set_params(alpha=0.1, beta=1, fit_linear=True)

    import scipy.sparse as sp

    fm.fit(sp.vstack([X, X]), np.concatenate([y, y]))
    y_val_pred = fm.predict(X_val)
    print("FM rank={}, validation RMSE={:.2f}".format(
        fm.rank_,
        np.sqrt(mean_squared_error(y_val, y_val_pred))))

    fm.set_params(beta=1.)
    fm.fit(X, y, sample_weight=2 * np.ones_like(y))
    y_val_pred = fm.predict(X_val)
    print("FM rank={}, validation RMSE={:.2f}".format(
        fm.rank_,
        np.sqrt(mean_squared_error(y_val, y_val_pred))))