import datetime
import os
import time
from os.path import expanduser, join, exists
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition.sparse_pca import IncrementalSparsePCA
from sklearn import datasets
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.utils import check_random_state


def single_run(estimator, data):
    print('        ')
    for i in range(30):
        print('New epoch %i' % i)
        this_data = data.copy()
        this_data -= np.mean(this_data, axis=0)
        this_data /= np.std(this_data, axis=0)
        estimator.partial_fit(this_data, deprecated=False)
    compute_time = estimator.debug_info_['total_time']
    this_data = this_data[:3]
    code = estimator.transform(this_data)
    reconstruction = code.dot(estimator.components_)
    estimator.subsets_ = None
    return estimator, reconstruction, compute_time


def run():
    faces = datasets.fetch_olivetti_faces()

    data = faces.images.reshape((faces.images.shape[0], -1))

    ###############################################################################
    # Learn the dictionary of images

    print('Learning the dictionary... ')
    rng = check_random_state(30)
    estimators = []
    warmup_estimator = IncrementalSparsePCA(n_components=30, alpha=0.01,
                                         n_iter=100000,
                                         random_state=rng,
                                         batch_size=20,
                                         debug_info=True,
                                         support=False,
                                         feature_ratio=1)
    for i in range(10):
        print('Epoch %i' % i)
        this_data = data.copy()
        this_data -= np.mean(this_data, axis=0)
        this_data /= np.std(this_data, axis=0)
        warmup_estimator.partial_fit(this_data, deprecated=False)
    dict_init = warmup_estimator.components_
    for support in [False, True]:
        estimators.append(IncrementalSparsePCA(n_components=30, alpha=0.01,
                                               n_iter=100000,
                                               random_state=rng,
                                               batch_size=20,
                                               debug_info=True,
                                               dict_init=dict_init,
                                               support=support,
                                               feature_ratio=10))
    # estimators.append(IncrementalSparsePCA(n_components=30, alpha=0.01,
    #                                        n_iter=100000,
    #                                        random_state=rng,
    #                                        batch_size=20,
    #                                        debug_info=True,
    #                                        dict_init=dict_init,
    #                                        support=True,
    #                                        feature_ratio=10))
    t0 = time.time()

    mem = Memory(cachedir=expanduser('~/sklearn_cache'), verbose=10)
    cached_single_run = mem.cache(single_run)

    res = Parallel(n_jobs=2, verbose=10)(delayed(cached_single_run)(
        estimator, data) for estimator in estimators)
    estimators, reconstructions, compute_times = zip(*res)
    dt = time.time() - t0
    print(compute_times)
    print('done in %.2fs.' % dt)

    output_dir = expanduser('~/output/incr_spca')
    output_dir = ²join(output_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                       '-%M-%S'))
    if not exists(output_dir):
        os.makedirs(output_dir)

    ###########################################################################
    # Plot the results
    fig = plt.figure()
    for j, estimator in enumerate(estimators):
        for i, component in enumerate(estimator.components_[:3]):
            if np.sum(component > 0) < np.sum(component < 0):
                component *= -1
            ax = fig.add_subplot(len(estimators), 4, 4 * j + i + 1)
            ax.imshow(component.reshape(faces.images[0].shape),
                      cmap=plt.cm.gray,
                      interpolation='nearest')
            ax.set_xticks(())
            ax.set_yticks(())
        ax = fig.add_subplot(len(estimators), 4, 4 * j + 4)
        ax.text(.5, .5,
                'ratio: %.2f' % estimator.feature_ratio,
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.axis('off')
    plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.5)
    plt.savefig(
        expanduser(join(output_dir, 'components.pdf')))

    fig = plt.figure()
    for j, (estimator, reconstruction) in enumerate(zip(estimators,
                                                        reconstructions)):
        for i, img in enumerate(reconstruction[:3]):
            if np.sum(component > 0) < np.sum(component < 0):
                component *= -1
            ax = fig.add_subplot(len(estimators), 4, 4 * j + i + 1)
            plt.imshow(img.reshape(faces.images[0].shape), cmap=plt.cm.gray,
                       interpolation='nearest')
            ax.set_xticks(())
            ax.set_yticks(())
        ax = fig.add_subplot(len(estimators), 4, 4 * j + 4)
        ax.text(.5, .5,
                'ratio: %.2f' % estimator.feature_ratio,
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.axis('off')
    plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.5)
    plt.savefig(join(output_dir, 'reconstruction.pdf'))


    plt.figure()
    for estimator, compute_time in zip(estimators, compute_times):
        residuals = estimator.debug_info_['residuals']
        plt.plot(np.linspace(0, 1, len(residuals)), residuals,
                 label='ratio %.2f' % estimator.feature_ratio)
        plt.xlabel('Time (s)')
        plt.ylabel('Objective value')
        plt.ylim([0, 2500])
        plt.legend(ncol=2)
    plt.tight_layout(pad=0.4)
    plt.savefig(expanduser(join(output_dir, 'residuals.pdf')))


if __name__ == '__main__':
    run()
