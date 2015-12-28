import json
from os.path import join

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.sparsefuncs import mean_variance_axis

from .base import csr_center_data, BaseRecommender, csr_mean_col


def _find_decomposition(X_ref, dict_learning,
                        n_epochs):
    # print('Not cached')
    # return (0, np.zeros(X_ref.shape[0]),
    #         np.zeros(X_ref.shape[1]),
    #         np.zeros((dict_learning.n_components,
    #                   X_ref.shape[1])),
    #         np.zeros((X_ref.shape[0], dict_learning.n_components)))

    # random_state = check_random_state(random_state)
    X_csr = X_ref.copy()
    interaction = csr_matrix((np.zeros_like(X_csr.data),
                              X_csr.indices, X_csr.indptr),
                             shape=X_csr.shape)
    for i in range(n_epochs):
        X_ref.data -= interaction.data
        (X_csr, global_mean,
         sample_mean, feature_mean) = csr_center_data(X_ref)
        X_ref.data += interaction.data
        X_csr.data += interaction.data

        # permutation = random_state.permutation(X_csr.shape[0])
        dict_learning.partial_fit(X_csr, deprecated=False)

        dictionary = dict_learning.components_
        code = dict_learning.transform(X_csr)

        # FIXME could be factored
        for j in range(X_csr.shape[0]):
            indices = X_csr.indices[X_csr.indptr[j]:X_csr.indptr[j + 1]]
            interaction.data[X_csr.indptr[j]:X_csr.indptr[j + 1]] = \
                code[j].dot(dictionary[:, indices])

        A, B, residual_stat = dict_learning.inner_stats_
        (last_cost, norm_cost, penalty_cost, n_seen_samples,
         count_seen_features, A_ref, B_ref) = residual_stat
        n_seen_samples = 0
        count_seen_features[:] = 0
        residual_stats = (last_cost, norm_cost, penalty_cost,
                          n_seen_samples,
                          count_seen_features, A_ref, B_ref)
        dict_learning.inner_stats_ = A, B, residual_stats
    return global_mean, sample_mean, feature_mean, dictionary, code


class DLRecommender(BaseRecommender):
    def __init__(self, fm_decoder=None,
                 random_state=None, n_components=10,
                 alpha=1., l1_ratio=0., algorithm='ridge',
                 n_epochs=1, batch_size=10,
                 learning_rate=0.5,
                 decreasing_batch_size=True,
                 fit_intercept=False,
                 debug_folder=None,
                 ):
        BaseRecommender.__init__(self, fm_decoder)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.n_components = n_components
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.fit_intercept = fit_intercept
        self.debug_folder = debug_folder
        self.decreasing_batch_size = decreasing_batch_size

    def _predict_quadratic(self, X_csr, samples, features):
        for i in range(X_csr.shape[0]):
            indices = X_csr.indices[X_csr.indptr[i]:X_csr.indptr[i + 1]]
            X_csr.data[X_csr.indptr[i]:
            X_csr.indptr[i + 1]] += self.code_[i].dot(
                    self.dictionary_[:, indices])

    def fit(self, X, y, **dump_kwargs):
        if self.debug_folder is not None:
            self.dump_init()
        X_ref = self.fm_decoder.fm_to_csr(X, y)
        n_iter = X_ref.shape[0] * self.n_epochs // self.batch_size

        random_state = check_random_state(self.random_state)
        dict_init = random_state.randn(self.n_components, X_ref.shape[1])

        dict_learning = MiniBatchDictionaryLearning(
                n_components=self.n_components,
                alpha=self.alpha,
                transform_alpha=self.alpha,
                fit_algorithm=self.algorithm,
                transform_algorithm=self.algorithm,
                dict_init=dict_init,
                l1_ratio=self.l1_ratio,
                batch_size=self.batch_size,
                shuffle=True,
                fit_intercept=self.fit_intercept,
                n_iter=n_iter,
                missing_values=0,
                learning_rate=self.learning_rate,
                verbose=3,
                debug_info=self.debug_folder is not None,
                random_state=self.random_state)

        if self.fit_intercept:
            self.dictionary_ = np.r_[np.ones((1, dict_init.shape[1])),
                                     dict_init]
            self.code_ = np.zeros((X.shape[0], self.n_components + 1))
        else:
            self.dictionary_ = dict_init
            self.code_ = np.zeros((X.shape[0], self.n_components))

        if self.debug_folder is None:
            (X_csr, self.global_mean_,
             self.sample_mean_, self.feature_mean_) = csr_center_data(X_ref)
            for i in range(self.n_epochs):
                dict_learning.partial_fit(X_csr, deprecated=False)
                if self.decreasing_batch_size:
                    dict_learning.set_params(batch_size=
                                             dict_learning.batch_size // 2)
            self.n_iter_ = dict_learning.n_iter_
            self.dictionary_ = dict_learning.components_
            self.code_ = dict_learning.transform(X_csr)

        if self.debug_folder is not None:
            (X_csr, self.global_mean_,
             self.sample_mean_, self.feature_mean_) = csr_center_data(X_ref)
            self.dump_inter(**dump_kwargs)

            for i in range(self.n_epochs):
                permutation = random_state.permutation(X_csr.shape[0])

                batches = gen_batches(X_csr.shape[0],
                                      X_csr.shape[0] // 5 + 1)
                last_seen = 0
                for batch in batches:
                    last_seen = max(batch.stop, last_seen)
                    dict_learning.partial_fit(X_csr[permutation[batch]],
                                              deprecated=False)
                    self.dictionary_ = dict_learning.components_
                    self.code_[permutation[:last_seen]] = dict_learning.\
                        transform(X_csr[permutation[:last_seen]])
                    self.n_iter_ = dict_learning.n_iter_
                    self.dump_inter(debug_dict=dict_learning.debug_info_,
                                    **dump_kwargs)
                    if self.decreasing_batch_size:
                        dict_learning.set_params(batch_size=
                                                 dict_learning.batch_size // 2)
            self.dictionary_ = dict_learning.components_
            self.code_ = dict_learning.transform(X_csr)
        return self

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
        probe_score[0] = self.n_iter_ if hasattr(self, 'n_iter_') else 0
        for i, (X, y) in enumerate(probe_list):
            probe_score[i + 1] = self.score(X, y)
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
        if len(probe_score) > 1:
            results['test_score'] = probe_score[1]
        if len(probe_score) > 2:
            results['train_score'] = probe_score[2]
        with open(join(self.debug_folder, 'results.json'), 'w+') as f:
            json.dump(results, f)

        if debug_dict is not None:
            residuals = debug_dict['residuals']
            density = debug_dict['density']
            values = debug_dict['values']
            np.save(join(self.debug_folder, 'residuals'), residuals)
            np.save(join(self.debug_folder, 'density'), density)
            np.save(join(self.debug_folder, 'values'), values)

        draw_stats(self.debug_folder)


def draw_stats(debug_folder):
    import matplotlib.pyplot as plt
    try:
        residuals = np.load(join(debug_folder, 'residuals.npy'))
        density = np.load(join(debug_folder, 'density.npy'))
        values = np.load(join(debug_folder, 'values.npy'))

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
    except:
        pass

    probe_score = np.load(join(debug_folder, 'probe_score.npy'))
    fig = plt.figure()
    plt.plot(probe_score[:, 0], probe_score[:, 1:], marker='o')
    plt.savefig(join(debug_folder, 'probe_score.pdf'))
    plt.close(fig)

    plt.close('all')
