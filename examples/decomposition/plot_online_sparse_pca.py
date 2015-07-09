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
from sklearn.decomposition.dict_learning import sparse_encode, MiniBatchDictionaryLearning

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
dict_learning = MiniBatchDictionaryLearning(n_components=n_components, alpha=0.,
                                            n_iter=1000, batch_size=10,
                                            fit_algorithm='ridge',
                                            transform_algorithm='ridge',
                                            transform_alpha=0.,
                                            tol=1e-4,
                                            verbose=10,
                                            l1_ratio=0.1,
                                            random_state=rng,
                                            n_jobs=3,
                                            debug_info=True)

###############################################################################
# Plot a sample of the input data

plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

###############################################################################
# Do the estimation and plot it
name = "Online Dictionary learning"
print("Extracting the top %d %s..." % (n_components, name))
t0 = time()
data = faces
dict_learning.fit(faces_centered)
train_time = (time() - t0)
print("done in %0.3fs" % train_time)
plot_gallery('%s - Train time %.1fs' % (name, train_time),
             dict_learning.components_[:n_components])

np.save('values', dict_learning.values_)
np.save('density', dict_learning.density_)
np.save('residuals', dict_learning.residuals_, )

print("%s - Component density" % name)
print 1 - np.sum(dict_learning.components_ == 0) / float(np.size(dict_learning.components_))
print("Code density")
code = dict_learning.transform(data)
print 1 - np.sum(code == 0) / float(np.size(code))
plot_gallery('%s - Reconstruction' % name,
             code[:n_components].dot(dict_learning.components_))

plt.show()
plt.close()
