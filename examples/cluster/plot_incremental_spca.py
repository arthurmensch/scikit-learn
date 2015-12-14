import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition.sparse_pca import IncrementalSparsePCA
from sklearn import datasets

import matplotlib
matplotlib.use('Qt4Agg')

def run():
    faces = datasets.fetch_olivetti_faces()

    data = faces.images.reshape((faces.images.shape[0], -1))

    ###############################################################################
    # Learn the dictionary of images

    print('Learning the dictionary... ')
    rng = 0
    incr_spca = IncrementalSparsePCA(n_components=30, alpha=0.01,
                                     n_iter=100000,
                                     random_state=rng, verbose=2,
                                     batch_size=20,
                                     debug_info=True,
                                     feature_ratio=7,
                                     )

    t0 = time.time()


    for i in range(20):
        print('Epoch %i' % i)
        this_data = data
        this_data -= np.mean(this_data, axis=0)
        this_data /= np.std(this_data, axis=0)
        incr_spca.partial_fit(this_data, deprecated=False)

    dt = time.time() - t0
    print('done in %.2fs.' % dt)

    ###############################################################################
    # Plot the results
    plt.figure(figsize=(4.2, 4))
    for i, component in enumerate(incr_spca.components_):
        if np.sum(component > 0) < np.sum(component < 0):
            component *= -1
        plt.subplot(10, 3, i + 1)
        plt.imshow(component.reshape(faces.images[0].shape), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('Patches of faces\nTrain time %.1fs on %d patches' %
                 (dt, 10 * len(faces.images)), fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    n_imgs = 10

    this_data = data[:10]
    this_data -= np.mean(data, axis=0)
    this_data /= np.std(data, axis=0)
    code = incr_spca.transform(this_data)

    reconstruction = code.dot(incr_spca.components_)
    # reconstruction *= np.std(this_data, axis=0)
    # reconstruction += np.std(this_data, axis=0)

    plt.figure(figsize=(4.2, 4))
    for i, (reconstructed_img, img) in enumerate(zip(reconstruction, this_data)):
        plt.subplot(n_imgs, 2, 2 * i + 1)
        plt.imshow(img.reshape(faces.images[0].shape), cmap=plt.cm.gray,
               interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        plt.subplot(n_imgs, 2, 2 * i + 2)
        plt.imshow(reconstructed_img.reshape(faces.images[0].shape), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    residuals = incr_spca.debug_info_['residuals']

    plt.figure(figsize=(4.2, 4))
    plt.plot(np.arange(len(residuals)), residuals, label='Residuals')
    plt.plot(np.arange(len(residuals)), incr_spca.debug_info_['norm_cost'], label='Norm cost')
    plt.plot(np.arange(len(residuals)), incr_spca.debug_info_['objective_cost'], label='Objective cost')
    plt.plot(np.arange(len(residuals)), incr_spca.debug_info_['penalty_cost'], label='Penalty cost')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run()