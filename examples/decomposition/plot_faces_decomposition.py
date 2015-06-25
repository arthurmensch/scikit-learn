"""
============================
Faces dataset decompositions
============================

This example applies to :ref:`olivetti_faces` different unsupervised
matrix decomposition (dimension reduction) methods from the module
:py:mod:`sklearn.decomposition` (see the documentation chapter
:ref:`decompositions`) .

"""
print(__doc__)

# Authors: Vlad Niculae, Alexandre Gramfort
# License: BSD 3 clause

import logging
from time import time

from numpy.random import RandomState
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition
from sklearn.decomposition.dict_learning import sparse_encode

import numpy as np

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 5, 3
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

###############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

l1_ratio = 0.001
alpha = 0.2

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

###############################################################################
# List of the different estimators, whether to center and transpose the
# problem, and whether the transformer uses the clustering API.
estimators = [
    # ('Eigenfaces - RandomizedPCA',
    #  decomposition.RandomizedPCA(n_components=n_components, whiten=True),
    #  True),
    #
    # ('Non-negative components - NMF',
    #  decomposition.NMF(n_components=n_components, init='nndsvda', beta=5.0,
    #                    tol=5e-3, sparseness='components'),
    #  False),
    #
    # ('Independent components - FastICA',
    #  decomposition.FastICA(n_components=n_components, whiten=True),
    #  True),
    #
    # ('Sparse comp. - MiniBatchSparsePCA',
    #  decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.6,
    #                                   n_iter=50, batch_size=3,
    #                                   random_state=rng),
    #  True),
    #
    # ('MiniBatchDictionaryLearning',
    #     decomposition.MiniBatchDictionaryLearning(n_components=n_components, alpha=0.1,
    #                                               n_iter=50, batch_size=10,
    #                                               random_state=rng),
    #  True),
    #
    # ('Cluster centers - MiniBatchKMeans',
    #     MiniBatchKMeans(n_clusters=n_components, tol=1e-3, batch_size=20,
    #                     max_iter=50, random_state=rng),
    #  True),
    #
    # ('Factor Analysis components - FA',
    #  decomposition.FactorAnalysis(n_components=n_components, max_iter=2),
    #  True),

    ('Sparse comp. - MiniBatchDictionaryLearning',
     decomposition.MiniBatchDictionaryLearning(n_components=n_components, alpha=0.1,
                                               n_iter=40, batch_size=10,
                                               fit_algorithm='cd',
                                               fit_update_dict_dir='feature',
                                               tol=1e-4,
                                               verbose=10,
                                               l1_ratio=0.03,
                                               random_state=rng,
                                               n_jobs=5,
                                               debug_info=True),
     True),
]


###############################################################################
# Plot a sample of the input data

plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

###############################################################################
# Do the estimation and plot it
for i, (name, estimator, center) in enumerate(estimators):
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()
    data = faces
    if center:
        data = faces_centered
    estimator.fit(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_
    if hasattr(estimator, 'noise_variance_'):
        plot_gallery("Pixelwise variance",
                     estimator.noise_variance_.reshape(1, -1), n_col=1,
                     n_row=1)
    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                 components_[:n_components])

    # XXX: To be removed
    if estimator.debug_info:
        np.save('/media/data/work/debug_drop/values_%i' % i, estimator.values_)
        np.save('/media/data/work/debug_drop/residuals_%i' % i, estimator.residuals_)
        np.save('/media/data/work/debug_drop/density_%i' % i, estimator.density_)

    if name == 'Sparse comp. - MiniBatchDictionaryLearning':
        print("%s - Component density" % name)
        print 1 - np.sum(components_ == 0) / float(np.size(components_))
        print("%s - Component density" % name)
        code = sparse_encode(data, components_, algorithm='lasso_cd', alpha=alpha)
        print 1 - np.sum(code == 0) / float(np.size(code))
        plot_gallery('%s - Reconstruction' % name,
                     code[:n_components].dot(components_))

plt.show()
plt.close()
