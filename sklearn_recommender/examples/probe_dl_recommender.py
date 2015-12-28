import datetime
import os
from os.path import join, expanduser

import numpy as np

from sklearn.externals.joblib import Parallel, delayed, Memory, dump
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state
from sklearn_recommender import DLRecommender
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
        # estimator.set_params(debug_folder=debug_folder)
        # estimator.fit(X_train, y_train, probe_list=[(X_test, y_test)])
        estimator.fit(X_train, y_train)
    else:
        estimator.fit(X_train, y_train)
    y_hat = estimator.predict(X_test)
    score = np.sqrt(mean_squared_error(y_hat, y_test))
    print('RMSE %s: %.3f' % (estimator, score))

    if output_dir is not None:
        with open(join(debug_folder, 'score'), 'w+') as f:
            f.write('score : %.4f' % score)
        dump(estimator, join(debug_folder, 'estimator'), compress=9)

    return score


output_dir = expanduser(join('~/output/dl_recommender/',
                             datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                              '-%M-%S'))
                        )
os.makedirs(output_dir)

random_state = check_random_state(0)
mem = Memory(cachedir=expanduser("~/cache"), verbose=10)
X_csr = mem.cache(fetch_ml_10m)(expanduser('~/data/own/ml-10M100K'),
                                remove_empty=True)

permutation = random_state.permutation(X_csr.shape[0])

X_csr = X_csr[permutation]

X, y = array_to_fm_format(X_csr)

uniform_split = ShuffleSplit(n_iter=1,
                             test_size=.25, random_state=random_state)

fm_decoder = FMDecoder(n_samples=X_csr.shape[0], n_features=X_csr.shape[1])

base_estimator = BaseRecommender(fm_decoder)

dl_list = [DLRecommender(fm_decoder,
                         n_components=n_components,
                         batch_size=batch_size,
                         n_epochs=4,
                         alpha=alpha,
                         learning_rate=learning_rate,
                         decreasing_batch_size=decreasing_batch_size,
                         fit_intercept=True,
                         l1_ratio=0.,
                         random_state=0)
           for alpha in np.logspace(-5, 0, 6)
           for learning_rate in np.linspace(.5, 1., 5)
           for n_components in [20]
           for batch_size, decreasing_batch_size in [[32, True]]]
estimators = dl_list

# convex_fm = ConvexFM(alpha=1e-9, beta=1, fit_linear=True, random_state=0,
#                      max_rank=20)
# convex_fm_cv = GridSearchCV(convex_fm, param_grid={'beta': np.logspace(-4, 0, 5)},
#                      cv=KFold(shuffle=False, n_folds=3),
#                      error_score=-1000,
#                      memory=mem,
#                      n_jobs=15,
#                      refit=True,
#                      verbose=10)
# estimators = [convex_fm_cv]

# estimators = [base_estimator]

scores = Parallel(n_jobs=12, verbose=10, max_nbytes='100M')(
        delayed(single_run)(X, y, estimator, train, test,
                            estimator_idx, split_idx,
                            output_dir=output_dir
                            )
        for split_idx, (train, test) in enumerate(
                uniform_split.split(X, y))
        for estimator_idx, estimator in enumerate(estimators))

print(scores)