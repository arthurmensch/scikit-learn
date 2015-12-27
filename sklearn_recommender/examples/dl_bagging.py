import datetime
import os
from os.path import join, expanduser

import numpy as np

from sklearn.externals.joblib import Parallel, delayed, Memory, dump, load
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit, KFold, GridSearchCV
from sklearn.utils import check_random_state
from sklearn_recommender import DLRecommender, ConvexFM
from sklearn_recommender.base import array_to_fm_format, FMDecoder, \
    BaseRecommender
from sklearn_recommender.datasets import fetch_ml_10m


def single_run(X, y,
               estimator, train, test,
               estimator_idx, split_idx,
               output_dir=None):
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    if output_dir is not None:
        debug_folder = join(output_dir, "split_{}_est_{}".format(split_idx,
                                                                 estimator_idx))
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)

    estimator.fit(X_train, y_train)
    # print(estimator.best_estimator_)
    # print(estimator.grid_scores_)
    y_hat = estimator.predict(X_test)
    score = np.sqrt(mean_squared_error(y_hat, y_test))
    print('RMSE %s: %.3f' % (estimator, score))

    if output_dir is not None:
        with open(join(debug_folder, 'score'), 'w+') as f:
            f.write('score : %.4f' % score)
        dump(estimator, join(debug_folder, 'estimator'))

    return score


output_dir = expanduser(join('~/output/dl_recommender/',
                             datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                              '-%M-%S'))
                        )
os.makedirs(output_dir)

random_state = check_random_state(0)
mem = Memory(cachedir=expanduser("~/cache"), verbose=10)
X_csr = mem.cache(fetch_ml_10m)(expanduser('~/data/own/ml-10M100K'),
                                remove_empty=True)  #, n_users=10000)

permutation = random_state.permutation(X_csr.shape[0])

X_csr = X_csr[permutation]

X, y = array_to_fm_format(X_csr)

uniform_split = ShuffleSplit(n_iter=4,
                             test_size=.25, random_state=random_state)

fm_decoder = FMDecoder(n_samples=X_csr.shape[0], n_features=X_csr.shape[1])

base_estimator = BaseRecommender(fm_decoder)

convex_fm = ConvexFM(fit_linear=True, alpha=0, max_rank=20,
                     beta=1, verbose=100)
dl_rec = DLRecommender(fm_decoder,
                       n_components=50,
                       batch_size=32,
                       n_epochs=4,
                       alpha=0.01,
                       learning_rate=1,
                       fit_intercept=True,
                       l1_ratio=0.,
                       random_state=0)

dl_cv = GridSearchCV(dl_rec, param_grid={'alpha': np.logspace(-3, 0, 4),
                                         'learning_rate':
                                             np.linspace(.5, 1, 5)},
                     cv=KFold(shuffle=False, n_folds=3),
                     error_score=-1000,
                     memory=mem,
                     n_jobs=30,
                     refit=True,
                     verbose=10)

scores = Parallel(n_jobs=1, verbose=10, max_nbytes='100M')(
        delayed(single_run)(X, y, dl_cv, train, test,
                            0, split_idx,
                            output_dir=output_dir
                            )
        for split_idx, (train, test) in enumerate(
                uniform_split.split(X, y)))
