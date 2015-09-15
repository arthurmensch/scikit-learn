"""
Benchmarks of Lasso vs LassoLars

First, we fix a training set and increase the number of
samples. Then we plot the computation time as function of
the number of samples.

In the second benchmark, we increase the number of dimensions of the
training set. Then we plot the computation time as function of
the number of dimensions.

In both cases, only 10% of the features are informative.
"""
import gc
from time import time
from nose.tools import assert_less
import numpy as np
from sklearn.utils import check_array

from sklearn.datasets.samples_generator import make_regression


def compute_bench(alpha, n_samples, n_features, precompute):
    lasso_results = []
    lasso_screening_results = []
    lars_lasso_results = []

    it = 0

    for ns in n_samples:
        for nf in n_features:
            it += 1
            print('==================')
            print('Iteration %s of %s' % (it, max(len(n_samples),
                                          len(n_features))))
            print('==================')
            n_informative = nf // 10
            X, Y, coef_ = make_regression(n_samples=ns, n_features=nf,
                                          n_informative=n_informative,
                                          noise=0.1, coef=True)

            X /= np.sqrt(np.sum(X ** 2, axis=0))  # Normalize data
            X = check_array(X, order='F', dtype='float64')

            gc.collect()
            print("- benchmarking Lasso")
            lasso = Lasso(alpha=alpha, fit_intercept=False,
                        precompute=precompute, tol=1e-10)
            tstart = time()
            lasso.fit(X, Y, check_input=False)
            lasso_results.append(time() - tstart)

            gc.collect()
            print("- benchmarking Lasso with screening")
            lasso_screening = Lasso(alpha=alpha, fit_intercept=False,
                        precompute=precompute, screening=10,
                        tol=1e-10)
            tstart = time()
            lasso_screening.fit(X, Y, check_input=False)
            lasso_screening_results.append(time() - tstart)

            gc.collect()
            print("- benchmarking LassoLars")
            lars = LassoLars(alpha=alpha, fit_intercept=False, max_iter=2000,
                            normalize=False, precompute=precompute)
            tstart = time()
            lars.fit(X, Y)
            lars_lasso_results.append(time() - tstart)
            diff = np.sum((lasso_screening.coef_ - lars.coef_) ** 2)
            print(diff)
            assert_less(diff, 1e-10)

    return lasso_results, lasso_screening_results, lars_lasso_results


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

if __name__ == '__main__':
    from sklearn.linear_model import Lasso, LassoLars
    import matplotlib.pyplot as plt
    import matplotlib

    params = {'backend': 'Agg',
              'axes.labelsize': 6,
              'text.fontsize': 6} # extend as needed
    matplotlib.rcParams.update(params)

    alpha = 1  # regularization parameter

    n_features = 10
    list_n_samples = np.linspace(100, 1000000, 5).astype(np.int)
    lasso_results, lasso_screening_results, lars_lasso_results = \
        compute_bench(alpha, list_n_samples, [n_features], precompute=True)

    plt.figure('scikit-learn LASSO benchmark results', figsize=get_figsize(307.28987,  wf=1, hf=0.66))
    plt.subplot(121)
    plt.plot(np.array(list_n_samples) / 1000, lasso_results, 'b-', label='Lasso')
    plt.plot(np.array(list_n_samples) / 1000, lasso_screening_results, 'g-',
             label='Lasso (screening)')
    plt.plot(np.array(list_n_samples) / 1000, lars_lasso_results, 'r-', label='LassoLars')
    plt.title('precomputed Gram matrix, %d features'
              % (n_features))
    # plt.legend(loc='upper left')
    plt.xlabel('number of samples (x1000)')
    plt.ylabel('Time (s)')
    plt.axis('tight')

    n_samples = 2000
    list_n_features = np.linspace(500, 3000, 5).astype(np.int)
    lasso_results, lasso_screening_results, lars_lasso_results = \
        compute_bench(alpha, [n_samples], list_n_features, precompute=False)
    plt.subplot(122)
    plt.plot(list_n_features, lasso_results, 'b-', label='Lasso')
    plt.plot(list_n_features, lasso_screening_results, 'g-',
             label='Lasso (screening)')
    plt.plot(list_n_features, lars_lasso_results, 'r-', label='LassoLars')
    plt.title('%d samples' % (n_samples))
    plt.legend(loc='upper left')
    plt.xlabel('number of features')
    # plt.ylabel('Time (s)')
    plt.axis('tight')
    plt.savefig('/volatile/arthur/work/repos/internship/09_1015_presentation_sklearn_pjl/figures/bench_lasso.pdf')
    plt.show()
