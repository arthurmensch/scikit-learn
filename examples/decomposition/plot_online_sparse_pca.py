import logging
from math import sqrt
from time import time

from numpy.random import RandomState
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib

params = {'backend': 'Agg',
          'axes.labelsize': 6,
          'text.fontsize': 6} # extend as needed
matplotlib.rcParams.update(params)

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition.dict_learning import sparse_encode,\
    MiniBatchDictionaryLearning

import numpy as np

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 5, 6
n_components = n_row * n_col
# image_shape = (128, 64)
image_shape = (64, 64)
rng = RandomState(0)

###############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data
# faces = np.tile(faces, (1, 2))

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)

###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(1. * n_col, 1.13 * n_row))
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


def get_figsize(columnwidth, wf=0.5, hf=(5.**0.5-1.0)/2.0, ):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]

###############################################################################
# It is necessary to add regularisation to sparse encoder (either l1 or l2).
# XXX: This should be mentionned in the documentation
estimators = []
labels = ['cd 10e-4', 'cd 10e-8', 'lars']

estimators.append(MiniBatchDictionaryLearning(n_components=n_components,
                                            alpha=1,
                                            n_iter=4000, batch_size=10,
                                            fit_algorithm="cd",
                                            transform_algorithm="lasso_cd",
                                            transform_alpha=1,
                                            tol=1e-4,
                                            lasso_tol=1e-8,
                                            verbose=10,
                                            random_state=rng,
                                            n_jobs=1)
                  )
estimators.append(MiniBatchDictionaryLearning(n_components=n_components,
                                            alpha=1,
                                            n_iter=4000, batch_size=10,
                                            fit_algorithm="cd",
                                            transform_algorithm="lasso_cd",
                                            transform_alpha=1,
                                            tol=1e-4,
                                            lasso_tol=1e-4,
                                            verbose=10,
                                            random_state=rng,
                                            n_jobs=1)
                  )
estimators.append(MiniBatchDictionaryLearning(n_components=n_components,
                                            alpha=1,
                                            n_iter=4000, batch_size=10,
                                            fit_algorithm="lars",
                                            transform_algorithm="lasso_lars",
                                            transform_alpha=1,
                                            tol=1e-4,
                                            verbose=10,
                                            random_state=rng,
                                            n_jobs=1)
                  )

###############################################################################
# Plot a sample of the input data

# plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

# plt.savefig('faces.pdf')

###############################################################################
# Do the estimation and plot it
name = "Online Dictionary learning"
print("Extracting the top %d %s..." % (n_components, name))
# sparsity = np.zeros(11)
# for tile in range(1, 11):
tile = 1
data = np.tile(faces_centered, (1, tile))
image_shape = (image_shape[0] * tile, image_shape[1])

train_times = []
for estimator in estimators:
    t0 = time()
    estimator.fit(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    # plot_gallery('%s - Train time %.1fs' % (name, train_time),
    #          estimator.components_[:n_components])

    print("%s - Component density" % name)
    print(1 - np.sum(estimator.components_ == 0)\
              / float(np.size(estimator.components_)))
    print("Code density")
    code = estimator.transform(data)
    print 1 - np.sum(code == 0) / float(np.size(code))
    # plot_gallery('%s - Reconstruction' % name,
    #              code[:n_components].dot(estimator.components_))
    train_times.append(train_time)

fig = plt.figure(figsize=get_figsize(307.28987,  wf=1, hf=0.66))
for i, estimator in enumerate(estimators):
    plt.plot(estimator.residuals, estimator.times,
             label=labels[i])
plt.xlabel('Empirical objective')
plt.ylabel('Time')
# plt.xlim([28, 29.5])
# plt.ylim([0, 90])
plt.legend(loc='upper left')

# plt.savefig('lars_cd_perf.pdf')
plt.show()