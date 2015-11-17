import time
from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition.sparse_pca import IncrementalSparsePCA
from sklearn import datasets
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed, Memory


def single_run(estimator, data):
    for i in range(200):
        print('Epoch %i' % i)
        this_data = data.copy()
        this_data -= np.mean(this_data, axis=0)
        this_data /= np.std(this_data, axis=0)
        estimator.partial_fit(this_data, deprecated=False)
    first_time = estimator.debug_info_['total_time']
    feature_ratio = estimator.feature_ratio
    delattr(estimator, 'inner_stats_')
    estimator.set_params(feature_ratio=1)
    for i in range(200):
        print('New epoch %i' % i)
        this_data = data.copy()
        this_data -= np.mean(this_data, axis=0)
        this_data /= np.std(this_data, axis=0)
        estimator.partial_fit(this_data, deprecated=False)
    second_time = estimator.debug_info_['total_time'] - first_time
    estimator.set_params(feature_ratio=feature_ratio)
    this_data = this_data[:3]
    code = estimator.transform(this_data)
    reconstruction = code.dot(estimator.components_)
    estimator.subsets_ = None
    return estimator, reconstruction, (first_time, second_time)


def run():
    faces = datasets.fetch_olivetti_faces()

    data = faces.images.reshape((faces.images.shape[0], -1))

    ###############################################################################
    # Learn the dictionary of images

    print('Learning the dictionary... ')
    rng = 0
    estimators = []
    for feature_ratio in np.linspace(1, 10, 7):
        estimators.append(IncrementalSparsePCA(n_components=30, alpha=0.01,
                                               n_iter=100000,
                                               random_state=rng, verbose=2,
                                               batch_size=20,
                                               debug_info=True,
                                               feature_ratio=feature_ratio))

    t0 = time.time()

    mem = Memory(cachedir=expanduser('~/sklearn_cache'), verbose=10)
    cached_single_run = mem.cache(single_run)

    res = Parallel(n_jobs=7, verbose=10)(delayed(cached_single_run)(estimator, data) for estimator in estimators)
    estimators, reconstructions, compute_times = zip(*res)
    dt = time.time() - t0
    print(compute_times)
    print('done in %.2fs.' % dt)

    ###############################################################################
    # Plot the results
    plt.figure(figsize=(4.2, 4))
    for j, estimator in enumerate(estimators):
        for i, component in enumerate(estimator.components_[:3]):
            if np.sum(component > 0) < np.sum(component < 0):
                component *= -1
            plt.subplot(10, 3, 3 * j + i + 1)
            plt.imshow(component.reshape(faces.images[0].shape), cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())

    plt.suptitle('Patches of faces\nTrain time %.1fs on %d patches' %
                 (dt, 10 * len(faces.images)), fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.savefig(expanduser('~/output/incr_spca/components.pdf'))

    plt.figure(figsize=(4.2, 4))
    for j, (estimator, reconstruction) in enumerate(zip(estimators, reconstructions)):
        for i, img in enumerate(reconstruction):
            plt.subplot(10, 3, 3 * j + i + 1)
            plt.imshow(img.reshape(faces.images[0].shape), cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
    plt.savefig(expanduser('~/output/incr_spca/reconstruction.pdf'))

    plt.figure(figsize=(8, 8))
    for estimator, compute_time in zip(estimators, compute_times):
        residuals = estimator.debug_info_['residuals']
        plt.plot(np.concatenate((np.linspace(0, compute_time[0], len(residuals) // 2, endpoint=False),
                                 np.linspace(compute_time[0], compute_time[0] + compute_time[1],
                                             len(residuals) // 2))), residuals,
                 label='ratio %.2f' % estimator.feature_ratio)
        plt.scatter(compute_time[0], residuals[len(residuals) // 2 - 1])
        plt.legend()
    plt.savefig(expanduser('~/output/incr_spca/residuals.pdf'))


if __name__ == '__main__':
    run()
