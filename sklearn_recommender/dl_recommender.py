import json
from os.path import join

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.externals.joblib import Memory
from sklearn.utils import check_random_state, gen_batches

from .base import csr_center_data, BaseRecommender


def _find_decomposition(X_ref, dict_learning,
                        n_epochs, random_state):
    random_state = check_random_state(random_state)
    print(X_ref.indices.shape)
    print(X_ref.indptr.shape)
    print(X_ref.data)
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

        permutation = random_state.permutation(X_csr.shape[0])
        dict_learning.partial_fit(X_csr[permutation], deprecated=False)

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
                 memory=Memory(cachedir=None),
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
        self.memory = memory
        self.debug_folder = debug_folder

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
        dict_init = random_state.randn(self.n_components,
                                       X_ref.shape[1])
        dict_learning = MiniBatchDictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha,
            transform_alpha=self.alpha,
            fit_algorithm=self.algorithm,
            transform_algorithm=self.algorithm,
            dict_init=dict_init,
            l1_ratio=self.l1_ratio,
            batch_size=self.batch_size,
            shuffle=False,
            n_iter=n_iter,
            missing_values=0,
            learning_rate=self.learning_rate,
            verbose=3,
            debug_info=self.debug_folder is not None,
            random_state=self.random_state)

        if self.debug_folder is None:
            (self.global_mean_, self.sample_mean_,
             self.feature_mean_, self.dictionary_, self.code_) = \
                self.memory.cache(_find_decomposition)(X_ref, dict_learning,
                                    self.n_epochs, self.random_state)
        if self.debug_folder is not None:
            X_csr = X_ref.copy()
            interaction = csr_matrix((np.empty_like(X_csr.data),
                          X_csr.indices, X_csr.indptr),
                         shape=X_csr.shape)
            self.code_ = np.zeros((X.shape[0], self.n_components))
            for i in range(self.n_epochs):
                X_ref.data -= interaction.data
                (X_csr, self.global_mean_,
                 self.sample_mean_, self.feature_mean_) = csr_center_data(X_ref)
                X_ref.data += interaction.data
                X_csr.data += interaction.data
                permutation = random_state.permutation(X_csr.shape[0])

                batches = gen_batches(X_csr.shape[0],
                                      X_csr.shape[0] // 5)
                last_seen = 0
                for batch in batches:
                    last_seen = max(batch.stop, last_seen)
                    dict_learning.partial_fit(X_csr[permutation[batch]],
                                              deprecated=False)
                    self.dictionary_ = dict_learning.components_
                    self.code_[:last_seen] = dict_learning.transform(
                        X_csr[:last_seen])
                    self.n_iter_ = dict_learning.n_iter_
                    self.dump_inter(debug_dict=dict_learning.debug_info_,
                                    **dump_kwargs)

                dict_learning.partial_fit(X_csr[permutation], deprecated=False)

                self.dictionary_ = dict_learning.components_
                self.code_ = dict_learning.transform(X_csr)

                for j in range(X_csr.shape[0]):
                    indices = X_csr.indices[X_csr.indptr[j]:
                    X_csr.indptr[j + 1]]
                    interaction.data[X_csr.indptr[j]:X_csr.indptr[j + 1]] = \
                        self.code_[j].dot(self.dictionary_[:, indices])

                A, B, residual_stat = dict_learning.inner_stats_
                last_cost, norm_cost, penalty_cost, n_seen_samples, \
                count_seen_features, A_ref, B_ref = residual_stat
                n_seen_samples = 0
                count_seen_features[:] = 0
                residual_stats = (last_cost, norm_cost, penalty_cost,
                                  n_seen_samples,
                                  count_seen_features, A_ref, B_ref)
                dict_learning.inner_stats_ = A, B, residual_stats
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
        probe_score[0] = self.n_iter_
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
    residuals = np.load(join(debug_folder, 'residuals.npy'))
    density = np.load(join(debug_folder, 'density.npy'))
    values = np.load(join(debug_folder, 'values.npy'))
    probe_score = np.load(join(debug_folder, 'probe_score.npy'))

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
    plt.plot(probe_score[:, 0], probe_score[:, 1:], marker='o')
    plt.savefig(join(debug_folder, 'probe_score.pdf'))
    plt.close(fig)

    plt.close('all')