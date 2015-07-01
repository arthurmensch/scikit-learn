""" Dictionary learning
"""
from __future__ import print_function
# Author: Vlad Niculae, Gael Varoquaux, Alexandre Gramfort
# License: BSD 3 clause

import time
import sys
import itertools

from math import sqrt, ceil

import numpy as np
from scipy import linalg
from numpy.lib.stride_tricks import as_strided

from ..base import BaseEstimator, TransformerMixin
from ..externals.joblib import Parallel, delayed, cpu_count
from ..externals.six.moves import zip
from ..utils import (check_array, check_random_state, gen_even_slices,
                     gen_batches, _get_n_jobs)
from ..utils.extmath import randomized_svd, row_norms
from ..utils.validation import check_is_fitted
from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars, Ridge
from .dict_learning_fast import _update_dict_feature_wise_fast
from ..utils.enet_proj_fast import enet_projection, enet_norm

from multiprocessing.pool import ThreadPool


def _sparse_encode(X, dictionary, gram, cov=None, algorithm='lasso_lars',
                   regularization=None, copy_cov=True,
                   init=None, max_iter=1000,
                   random_state=None):
    """Generic sparse coding

    Each column of the result is the solution to a Lasso problem.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        Data matrix.

    dictionary: array of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows.

    gram: None | array, shape=(n_components, n_components)
        Precomputed Gram matrix, dictionary * dictionary'
        gram can be None if method is 'threshold'.

    cov: array, shape=(n_components, n_samples)
        Precomputed covariance, dictionary * X'

    algorithm: {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold', 'ridge'}
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than regularization
        from the projection dictionary * data'
        ols: uses a non-penalized least square fit (regularization parameter is ignored)

    regularization : int | float
        The regularization parameter. It corresponds to alpha when
        algorithm is 'lasso_lars', 'lasso_cd' or 'threshold'.
        Otherwise it corresponds to n_nonzero_coefs.

    init: array of shape (n_samples, n_components)
        Initialization value of the sparse code. Only used if
        `algorithm='lasso_cd'`.

    max_iter: int, 1000 by default
        Maximum number of iterations to perform if `algorithm='lasso_cd'`.

    copy_cov: boolean, optional
        Whether to copy the precomputed covariance matrix; if False, it may be
        overwritten.

    Returns
    -------
    code: array of shape (n_components, n_features)
        The sparse codes

    See also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples, n_features = X.shape
    if cov is None and algorithm != 'lasso_cd':
        # overwriting cov is safe
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    if algorithm == 'lasso_lars':
        alpha = float(regularization) / n_features  # account for scaling
        try:
            err_mgt = np.seterr(all='ignore')
            lasso_lars = LassoLars(alpha=alpha, fit_intercept=False,
                                   verbose=False, normalize=False,
                                   precompute=gram, fit_path=False)
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lasso_lars.coef_
        finally:
            np.seterr(**err_mgt)

    elif algorithm == 'lasso_cd':
        alpha = float(regularization) / n_features  # account for scaling
        clf = Lasso(alpha=alpha, fit_intercept=False, precompute=gram,
                    max_iter=max_iter, selection='random', random_state=random_state, warm_start=True)
        clf.coef_ = init
        clf.fit(dictionary.T, X.T)
        new_code = clf.coef_

    elif algorithm == 'lars':
        try:
            err_mgt = np.seterr(all='ignore')
            lars = Lars(fit_intercept=False, verbose=False, normalize=False,
                        precompute=gram, n_nonzero_coefs=int(regularization),
                        fit_path=False)
            lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lars.coef_
        finally:
            np.seterr(**err_mgt)

    elif algorithm == 'threshold':
        new_code = ((np.sign(cov) *
                    np.maximum(np.abs(cov) - regularization, 0)).T)

    elif algorithm == 'omp':
        new_code = orthogonal_mp_gram(gram, cov, regularization, None,
                                      row_norms(X, squared=True),
                                      copy_Xy=copy_cov).T

    elif algorithm == 'ridge':
        alpha = float(regularization) / n_features  # account for scaling
        lr = Ridge(alpha=alpha, fit_intercept=False, normalize=False)
        lr.fit(dictionary.T, X.T)
        new_code = lr.coef_

    else:
        raise ValueError('Sparse coding method must be "lasso_lars" '
                         '"lasso_cd",  "lasso", "threshold", "ols" or "omp", got %s.'
                         % algorithm)
    return new_code


# XXX : could be moved to the linear_model module
def sparse_encode(X, dictionary, gram=None, cov=None, algorithm='lasso_lars',
                  n_nonzero_coefs=None, alpha=None, copy_cov=True, init=None,
                  max_iter=1000, n_jobs=1, pool=None,
                  random_state=None):
    """Sparse coding

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Read more in the :ref:`User Guide <SparseCoder>`.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        Data matrix

    dictionary: array of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

    gram: array, shape=(n_components, n_components)
        Precomputed Gram matrix, dictionary * dictionary'

    cov: array, shape=(n_components, n_samples)
        Precomputed covariance, dictionary' * X

    algorithm: {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold', 'ridge'}
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection dictionary * X'
        ols: uses a non-penalized least square fit (regularization parameter is ignored)

    n_nonzero_coefs: int, 0.1 * n_features by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case.

    alpha: float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threhold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.

    init: array of shape (n_samples, n_components)
        Initialization value of the sparse codes. Only used if
        `algorithm='lasso_cd'`.

    max_iter: int, 1000 by default
        Maximum number of iterations to perform if `algorithm='lasso_cd'`.

    copy_cov: boolean, optional
        Whether to copy the precomputed covariance matrix; if False, it may be
        overwritten.

    n_jobs: int, optional
        Number of parallel jobs to run.

    Returns
    -------
    code: array of shape (n_samples, n_components)
        The sparse codes

    See also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    """
    dictionary = check_array(dictionary)
    X = check_array(X)
    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]

    if gram is None and algorithm != 'threshold':
        gram = np.dot(dictionary, dictionary.T)
    if cov is None:
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    if algorithm in ('lars', 'omp'):
        regularization = n_nonzero_coefs
        if regularization is None:
            regularization = min(max(n_features / 10, 1), n_components)
    else:
        regularization = alpha
        if regularization is None:
            regularization = 1.

    if (pool is None and n_jobs == 1) or algorithm == 'threshold':
        return _sparse_encode(X, dictionary, gram, cov=cov,
                              algorithm=algorithm,
                              regularization=regularization, copy_cov=copy_cov,
                              init=init, max_iter=max_iter, random_state=random_state)

    # Enter parallel code block
    code = np.empty((n_samples, n_components))

    if pool is not None and algorithm is 'lasso_cd':
        random_state = check_random_state(random_state)
        slices = list(gen_even_slices(n_samples, pool._processes))
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_samples)
        code_views = pool.map(lambda this_slice_, this_seed: _sparse_encode(X[this_slice_], dictionary, gram, cov[:, this_slice_], algorithm,
                                                    regularization=regularization, copy_cov=copy_cov,
                                                    init=init[this_slice_] if init is not None else None,
                                                    max_iter=max_iter, random_state=this_seed),
                              zip(slices, seeds))
    else:
        slices = list(gen_even_slices(n_samples, _get_n_jobs(n_jobs)))
        code_views = Parallel(n_jobs=n_jobs, backend='threading' if algorithm == 'lasso_cd' else 'multiprocessing')(
            delayed(_sparse_encode)(
                X[this_slice], dictionary, gram, cov[:, this_slice], algorithm,
                regularization=regularization, copy_cov=copy_cov,
                init=init[this_slice] if init is not None else None,
                max_iter=max_iter)
            for this_slice in slices)
    for this_slice, this_view in zip(slices, code_views):
            code[this_slice] = this_view

    return code


def _update_dict(dictionary, Y, code, verbose=False, return_r2=False,
                                update_dict_dir='component',
                                l1_ratio=0.1, radius=1., online=False, random_state=None,
                                pool=None):
    """Update the dense dictionary factor in place, constraining dictionary component to have a unit l2 norm.

    Parameters
    ----------
    dictionary: array of shape (n_features, n_components)
        Value of the dictionary at the previous iteration.

    Y: array of shape (n_features, n_samples)
        Data matrix.

    code: array of shape (n_components, n_samples)
        Sparse coding of the data against which to optimize the dictionary.

    verbose:
        Degree of output the procedure will print.

    return_r2: bool
        Whether to compute and return the residual sum of squares corresponding
        to the computed solution.

    online: bool,
        Wether the update we perform is part of an online algorithm or not (this changes derivation of residuals
        and of step size)

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    pool : ThreadPool, optional
        thread pool to use

    Returns
    -------
    dictionary: array of shape (n_features, n_components)
        Updated dictionary.

    """

    n_components = len(code)
    component_range = random_state.permutation(n_components)
    n_features = Y.shape[0]
    random_state = check_random_state(random_state)
    # Residuals, computed 'in-place' for efficiency
    R = -np.dot(dictionary, code)
    R += Y

    if update_dict_dir == 'component':
        # radius *= n_components
        threshold = 1e-20 # * n_components
        R = np.asfortranarray(R)
        ger, = linalg.get_blas_funcs(('ger',), (dictionary, code))
        for k in component_range:
            # R <- 1.0 * U_k * V_k^T + R
            R = ger(1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
            if online:
                dictionary[:, k] = R[:, k]
            else:
                dictionary[:, k] = np.dot(R, code[k, :].T)
            # Scale k'th atom
            atom_norm_square = enet_norm(dictionary[:, k], l1_ratio=l1_ratio)
            if atom_norm_square < threshold:
                if verbose == 1:
                    sys.stdout.write("+")
                    sys.stdout.flush()
                elif verbose:
                    print("Adding new random atom")
                dictionary[:, k] = 10 * random_state.randn(n_features)
                # Setting corresponding coefs to 0
                code[k, :] = 0.0
                dictionary[:, k] = enet_projection(dictionary[:, k], radius=radius,
                                                   l1_ratio=l1_ratio)
            else:
                dictionary[:, k] = enet_projection(dictionary[:, k], radius=radius,
                                                   l1_ratio=l1_ratio)
                # R <- -1.0 * U_k * V_k^T + R
                R = ger(-1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
    # XXX: This is untested and EXPERIMENTAL
    elif update_dict_dir == 'feature':
        # radius *= n_features
        # We use a cython auxiliary function, because:
        # We enter a large loop with n_features iterations, that can be parallelized releasing the GIL
        if pool is None:
            _update_dict_feature_wise_fast(dictionary, R, code, l1_ratio, radius)
        else:
            slices = list(gen_even_slices(n_features, pool._processes))
            pool.map(lambda this_slice_: _update_dict_feature_wise_fast(dictionary[this_slice_], R[this_slice_],
                                                                        code, l1_ratio, radius),
                     slices)
    else:
        raise ValueError('Wrong argument for direction of dictionary update')

    if return_r2:
        if online:
            # Y = B_t in online setting
            # 1 / 2 Tr(D^T D A_t) - Tr(D^T B_t)
            R += Y
            residual = -np.sum(dictionary * R) / 2
        else:
            R **= 2
            # R is fortran-ordered. For numpy version < 1.6, sum does not
            # follow the quick striding first, and is thus inefficient on
            # fortran ordered data. We take a flat view of the data with no
            # striding
            R = as_strided(R, shape=(R.size, ), strides=(R.dtype.itemsize,))
            residual = np.sum(R)
        return dictionary, residual
    return dictionary


def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-8,
                  method='lars', n_jobs=1, dict_init=None, code_init=None,
                  callback=None, verbose=False, random_state=None,
                  return_n_iter=False):
    """Solves a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        Data matrix.

    n_components: int,
        Number of dictionary atoms to extract.

    alpha: int,
        Sparsity controlling parameter.

    max_iter: int,
        Maximum number of iterations to perform.

    tol: float,
        Tolerance for the stopping condition.

    method: {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs: int,
        Number of parallel jobs to run, or -1 to autodetect.

    dict_init: array of shape (n_components, n_features),
        Initial value for the dictionary for warm restart scenarios.

    code_init: array of shape (n_samples, n_components),
        Initial value for the sparse code for warm restart scenarios.

    callback:
        Callable that gets invoked every five iterations.

    verbose:
        Degree of output the procedure will print.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    return_n_iter : bool
        Whether or not to return the number of iterations.

    Returns
    -------
    code: array of shape (n_samples, n_components)
        The sparse code factor in the matrix factorization.

    dictionary: array of shape (n_components, n_features),
        The dictionary factor in the matrix factorization.

    errors: array
        Vector of errors at each iteration.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    See also
    --------
    dict_learning_online
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """
    if method not in ('lars', 'cd'):
        raise ValueError('Coding method %r not supported as a fit algorithm.'
                         % method)
    method = 'lasso_' + method

    t0 = time.time()
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Init the code and the dictionary with SVD of Y
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order='F')
        # Don't copy V, it will happen below
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    # Fortran-order dict, as we are going to access its row vectors
    dictionary = np.array(dictionary, order='F')

    residuals = 0

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print('[dict_learning]', end=' ')

    # If max_iter is 0, number of iterations returned should be zero
    ii = -1

    for ii in range(max_iter):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print ("Iteration % 3i "
                   "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                   % (ii, dt, dt / 60, current_cost))

        # Update code
        code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha,
                             init=code, n_jobs=n_jobs)
        # Update dictionary
        dictionary, residuals = _update_dict(dictionary.T, X.T, code.T,
                                             verbose=verbose, return_r2=True,
                                             online=False,
                                             random_state=random_state,
                                             l1_ratio=0.)
        dictionary = dictionary.T

        # Cost function
        current_cost = 0.5 * residuals + alpha * np.sum(np.abs(code))
        errors.append(current_cost)

        if ii > 0:
            dE = errors[-2] - errors[-1]
            # assert(dE >= -tol * errors[-1])
            if dE < tol * errors[-1]:
                if verbose == 1:
                    # A line return
                    print("")
                elif verbose:
                    print("--- Convergence reached after %d iterations" % ii)
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    if return_n_iter:
        return code, dictionary, errors, ii + 1
    else:
        return code, dictionary, errors


def dict_learning_online(X, n_components=2, alpha=1, l1_ratio=0.5, n_iter=100,
                         return_code=True, dict_init=None, callback=None,
                         batch_size=3, verbose=False, shuffle=True, n_jobs=1,
                         method='lars', update_dict_dir='component',
                         iter_offset=0, tol=1e-4,
                         random_state=None,
                         return_inner_stats=False, inner_stats=None,
                         return_n_iter=False, return_debug_info=False):
    """Solves a dictionary learning matrix factorization problem online.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                     with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. This is
    accomplished by repeatedly iterating over mini-batches by slicing
    the input data.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        Data matrix.

    n_components : int,
        Number of dictionary atoms to extract.

    alpha : float,
        Sparsity controlling parameter.

    n_iter : int,
        Number of iterations to perform.

    return_code : boolean,
        Whether to also return the code U or just the dictionary V.

    dict_init : array of shape (n_components, n_features),
        Initial value for the dictionary for warm restart scenarios.

    callback :
        Callable that gets invoked every five iterations.

    batch_size : int,
        The number of samples to take in each batch.

    verbose :
        Degree of output the procedure will print.

    shuffle : boolean,
        Whether to shuffle the data before splitting it in batches.

    n_jobs : int,
        Number of parallel jobs to run, or -1 to autodetect.

    method : {'lars', 'cd', 'ridge'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.
        ols: compute code using an ordinary least square method. alpha will be ignored

    update_dict_dir : {'component', 'feature'}
        Direction of convex set projection for dictionary update

    l1_ratio: float,
        Sparsity controlling parameter for dictionary projection. In [0, 1]

    tol: float,
        Stop controlling parameter

    iter_offset : int, default 0
        Number of previous iterations completed on the dictionary used for
        initialization.

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    return_inner_stats : boolean, optional
        Return the inner statistics A (dictionary covariance) and B
        (data approximation). Useful to restart the algorithm in an
        online setting. If return_inner_stats is True, return_code is
        ignored

    inner_stats : tuple of (A, B) ndarrays
        Inner sufficient statistics that are kept by the algorithm.
        Passing them at initialization is useful in online settings, to
        avoid loosing the history of the evolution.
        A (n_components, n_components) is the dictionary covariance matrix.
        B (n_features, n_components) is the data approximation matrix

    return_debug_info: bool,
        Whether to keep track of objective value, sparsity value and to keep a record of 10 dictionary trajectory

    return_n_iter : bool
        Whether or not to return the number of iterations.

    Returns
    -------
    code : array of shape (n_samples, n_components),
        the sparse code (only returned if `return_code=True`)

    dictionary : array of shape (n_components, n_features),
        the solutions to the dictionary learning problem

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to `True`.

    debug_info: tuple of (residuals, density, values),
        Debug Info

    See also
    --------
    dict_learning
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA

    """
    if n_components is None:
        n_components = X.shape[1]

    if update_dict_dir not in ('component', 'feature'):
        raise ValueError("Update dictionary direction mispecified")
    if method not in ('lars', 'cd', 'ridge'):
        raise ValueError('Coding method not supported as a fit algorithm.')
    if method in ('lars', 'cd'):
        method = 'lasso_' + method

    t0 = time.time()
    n_samples, n_features = X.shape
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Using a thread pool in the feature constraint setting
    if n_jobs > 1:
        pool = ThreadPool(n_jobs)
    else:
        pool = None

    # Init V with SVD of X
    if dict_init is not None:
        dictionary = dict_init
    else:
        _, S, dictionary = randomized_svd(X, n_components,
                                          random_state=random_state)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:
        dictionary = dictionary[:n_components, :]
    else:
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]
    dictionary = np.ascontiguousarray(dictionary.T)

    if verbose == 1:
        print('[dict_learning]', end=' ')

    if shuffle:
        X_train = X.copy()
        random_state.shuffle(X_train)
    else:
        X_train = X

    batches = gen_batches(n_samples, batch_size)
    batches = itertools.cycle(batches)

    # The covariance of the dictionary
    if inner_stats is None:
        A = np.zeros((n_components, n_components))
        # The data approximation
        B = np.zeros((n_features, n_components))
    else:
        A = inner_stats[0].copy()
        B = inner_stats[1].copy()

    # Residual variable for tolerance computation
    last_residual = np.iinfo(np.int32).max
    this_residual = 0
    penalty = 0
    patience = max(1, n_samples / (10*batch_size))
    this_patience = 0

    # If n_iter is zero, we need to return zero.
    ii = iter_offset - 1

    # XXX: To remove, for testing purpose only
    if return_debug_info:
        residuals = np.zeros(n_iter)
        density = np.zeros(n_iter)
        values = np.zeros((n_iter, 2))
        recorded_features = random_state.permutation(n_features)[:2]

    for ii, batch in zip(range(iter_offset, iter_offset + n_iter), batches):
        this_X = X_train[batch]
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            if verbose > 10 or ii % ceil(100. / verbose) == 0:
                print ("Iteration % 3i (elapsed time: % 3is, % 4.1fmn)"
                       % (ii, dt, dt / 60))
        # Setting n_jobs > 1 does not improve performance
        this_code = sparse_encode(this_X, dictionary.T, algorithm=method,
                                  alpha=alpha, n_jobs=1).T

        # Update the auxiliary variables
        if ii < batch_size - 1:
            theta = float((ii + 1) * batch_size)
        else:
            theta = float(batch_size ** 2 + ii + 1 - batch_size)
        beta = (theta + 1 - batch_size) / (theta + 1)

        A *= beta
        A += np.dot(this_code, this_code.T)
        B *= beta
        B += np.dot(this_X.T, this_code.T)

        # Update dictionary
        dictionary, this_residual = _update_dict(dictionary, B, A, verbose=verbose, l1_ratio=l1_ratio,
                                                 random_state=random_state, return_r2=True,
                                                 online=True, pool=pool
                                                 )
        this_residual /= 2
        if method in ('lars', 'cd'):
            penalty += alpha * np.sum(this_code)
        this_residual += penalty + np.sum(this_X ** 2) / 2
        this_residual /= iter_offset + (ii * batch_size) + 1
        change_ratio = abs(this_residual / last_residual - 1)
        if last_residual == 0:
            this_patience = patience
        else:
            if change_ratio < tol:
                this_patience += 1
            else:
                this_patience = 0
        if this_patience >= patience:
            break
        last_residual = this_residual

        # XXX: To be removed, for testing purpose only
        if return_debug_info:
            residuals[ii] = this_residual
            values[ii] = dictionary[recorded_features, 0] / sqrt(np.sum(dictionary[:, 0] ** 2))
            density[ii] = 1 - float(np.sum(dictionary == 0.)) / np.size(dictionary)

        # Maybe we need a stopping criteria based on the amount of
        # modification in the dictionary
        if callback is not None:
            callback(locals())

    if pool is not None:
        pool.close()

    # XXX: To remove, for testing purpose only
    if return_debug_info:
        debug_info = (residuals, density, values)
        return_inner_stats = True
        return_n_iter = True

    if return_inner_stats:
        if return_n_iter:
            # XXX: To remove
            if return_debug_info:
                return dictionary.T, (A, B), ii - iter_offset + 1, debug_info
            else:
                return dictionary.T, (A, B), ii - iter_offset + 1
        else:
            return dictionary.T, (A, B)
    if return_code:
        if verbose > 1:
            print('Learning code...', end=' ')
        elif verbose == 1:
            print('|', end=' ')
        code = sparse_encode(X, dictionary.T, algorithm=method, alpha=alpha,
                             n_jobs=1)
        if verbose > 1:
            dt = (time.time() - t0)
            print('done (total time: % 3is, % 4.1fmn)' % (dt, dt / 60))
        if return_n_iter:
            if debug_info:
                code, dictionary.T, ii - iter_offset + 1, debug_info
            else:
                return code, dictionary.T, ii - iter_offset + 1
        else:
            return code, dictionary.T

    if return_n_iter:
        return dictionary.T, ii - iter_offset + 1
    else:
        return dictionary.T


class SparseCodingMixin(TransformerMixin):
    """Sparse coding mixin"""

    def _set_sparse_coding_params(self, n_components,
                                  transform_algorithm='omp',
                                  transform_n_nonzero_coefs=None,
                                  transform_alpha=None, split_sign=False,
                                  n_jobs=1):
        self.n_components = n_components
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.split_sign = split_sign
        self.n_jobs = n_jobs

    def transform(self, X, y=None):
        """Encode the data as a sparse combination of the dictionary atoms.

        Coding method is determined by the object parameter
        `transform_algorithm`.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data

        """
        check_is_fitted(self, 'components_')

        # XXX : kwargs is not documented
        X = check_array(X)
        n_samples, n_features = X.shape

        code = sparse_encode(
            X, self.components_, algorithm=self.transform_algorithm,
            n_nonzero_coefs=self.transform_n_nonzero_coefs,
            alpha=self.transform_alpha, n_jobs=self.n_jobs)

        if self.split_sign:
            # feature vector is split into a positive and negative side
            n_samples, n_features = code.shape
            split_code = np.empty((n_samples, 2 * n_features))
            split_code[:, :n_features] = np.maximum(code, 0)
            split_code[:, n_features:] = -np.minimum(code, 0)
            code = split_code

        return code


class SparseCoder(BaseEstimator, SparseCodingMixin):
    """Sparse coding

    Finds a sparse representation of data against a fixed, precomputed
    dictionary.

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Read more in the :ref:`User Guide <SparseCoder>`.

    Parameters
    ----------
    dictionary : array, [n_components, n_features]
        The dictionary atoms used for sparse coding. Lines are assumed to be
        normalized to unit norm.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
    'threshold'}
        Algorithm used to transform the data:
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection ``dictionary * X'``

    transform_n_nonzero_coefs : int, ``0.1 * n_features`` by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case.

    transform_alpha : float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.

    split_sign : bool, False by default
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    n_jobs : int,
        number of parallel jobs to run

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        The unchanged dictionary atoms

    See also
    --------
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    sparse_encode
    """

    def __init__(self, dictionary, transform_algorithm='omp',
                 transform_n_nonzero_coefs=None, transform_alpha=None,
                 split_sign=False, n_jobs=1):
        self._set_sparse_coding_params(dictionary.shape[0],
                                       transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign, n_jobs)
        self.components_ = dictionary

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        return self


class DictionaryLearning(BaseEstimator, SparseCodingMixin):
    """Dictionary learning

    Finds a dictionary (a set of atoms) that can best be used to represent data
    using a sparse code.

    Solves the optimization problem::

        (U^*,V^*) = argmin 0.5 || Y - U V ||_2^2 + alpha * || U ||_1
                    (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int,
        number of dictionary elements to extract

    alpha : float,
        sparsity controlling parameter

    max_iter : int,
        maximum number of iterations to perform

    tol : float,
        tolerance for numerical error

    fit_algorithm : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
    'threshold'}
        Algorithm used to transform the data
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection ``dictionary * X'``

    transform_n_nonzero_coefs : int, ``0.1 * n_features`` by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case.

    transform_alpha : float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.

    split_sign : bool, False by default
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    n_jobs : int,
        number of parallel jobs to run

    code_init : array of shape (n_samples, n_components),
        initial value for the code, for warm restart

    dict_init : array of shape (n_components, n_features),
        initial values for the dictionary, for warm restart

    verbose :
        degree of verbosity of the printed output

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        dictionary atoms extracted from the data

    error_ : array
        vector of errors at each iteration

    n_iter_ : int
        Number of iterations run.

    Notes
    -----
    **References:**

    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
    for sparse coding (http://www.di.ens.fr/sierra/pdfs/icml09.pdf)

    See also
    --------
    SparseCoder
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """
    def __init__(self, n_components=None, alpha=1, max_iter=1000, tol=1e-8,
                 fit_algorithm='lars', transform_algorithm='omp',
                 transform_n_nonzero_coefs=None, transform_alpha=None,
                 n_jobs=1, code_init=None, dict_init=None, verbose=False,
                 split_sign=False, random_state=None):

        self._set_sparse_coding_params(n_components, transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign, n_jobs)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.code_init = code_init
        self.dict_init = dict_init
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self: object
            Returns the object itself
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        V, U, E, self.n_iter_ = dict_learning(
            X, n_components, self.alpha,
            tol=self.tol, max_iter=self.max_iter,
            method=self.fit_algorithm,
            n_jobs=self.n_jobs,
            code_init=self.code_init,
            dict_init=self.dict_init,
            verbose=self.verbose,
            random_state=random_state,
            return_n_iter=True)
        self.components_ = U
        self.error_ = E
        return self


class MiniBatchDictionaryLearning(BaseEstimator, SparseCodingMixin):
    """Mini-batch dictionary learning

    Finds a dictionary (a set of atoms) that can best be used to represent data
    using a sparse code.

    Solves the optimization problem::

       (U^*,V^*) = argmin 0.5 || Y - U V ||_2^2 + alpha * || U ||_1
                    (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int,
        number of dictionary elements to extract

    alpha : float,
        sparsity controlling parameter for code

    l1_ratio: float,
        Sparsity controlling parameter for dictionary components

    l1_ratio: float,
        sparsity controlling parameter, for 'elastic_net' method only

    tol: float,
        Tolerance for the stopping condition.

    n_iter : int,
        total number of iterations to perform

    fit_algorithm : {'lars', 'cd', 'ridge'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.
        elastic_net: Perform gradient-descent with elastic net projection
        to enforce dictionary sparsity, outputting non sparse code

    fit_update_dict_dir : {'component', 'feature'}
        Direction of convex set projection for dictionary update

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
    'threshold', 'ridge'}
        Algorithm used to transform the data.
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection dictionary * X'
        ols: uses a non-penalized least square fit (regularization parameter is ignored)

    transform_n_nonzero_coefs : int, ``0.1 * n_features`` by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case.

    transform_alpha : float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.

    split_sign : bool, False by default
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    n_jobs : int,
        number of parallel jobs to run

    dict_init : array of shape (n_components, n_features),
        initial value of the dictionary for warm restart scenarios

    verbose :
        degree of verbosity of the printed output

    batch_size : int,
        number of samples in each mini-batch

    shuffle : bool,
        whether to shuffle the samples before forming batches

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        components extracted from the data

    inner_stats_ : tuple of (A, B) ndarrays
        Internal sufficient statistics that are kept by the algorithm.
        Keeping them is useful in online settings, to avoid loosing the
        history of the evolution, but they shouldn't have any use for the
        end user.
        A (n_components, n_components) is the dictionary covariance matrix.
        B (n_features, n_components) is the data approximation matrix

    n_iter_ : int
        Number of iterations run.

    Notes
    -----
    **References:**

    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
    for sparse coding (http://www.di.ens.fr/sierra/pdfs/icml09.pdf)

    See also
    --------
    SparseCoder
    DictionaryLearning
    SparsePCA
    MiniBatchSparsePCA

    """
    def __init__(self, n_components=None, alpha=1, l1_ratio=0.5, n_iter=1000,
                 fit_algorithm='lars', fit_update_dict_dir='component', n_jobs=1, batch_size=3,
                 shuffle=True, dict_init=None, transform_algorithm='omp', tol=1e-4,
                 transform_n_nonzero_coefs=None, transform_alpha=None,
                 verbose=False, split_sign=False,
                 random_state=None,
                 debug_info=True):
        self._set_sparse_coding_params(n_components, transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign, n_jobs)
        self.alpha = alpha
        self.n_iter = n_iter
        self.fit_algorithm = fit_algorithm
        self.fit_update_dict_dir = fit_update_dict_dir
        self.dict_init = dict_init
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.split_sign = split_sign
        self.random_state = random_state
        self.l1_ratio = l1_ratio
        self.tol = tol
        # XXX: To remove
        self.debug_info = debug_info

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)

        U, (A, B), self.n_iter_, debug_info = dict_learning_online(
            X, self.n_components, self.alpha,
            n_iter=self.n_iter, return_code=False,
            method=self.fit_algorithm,
            update_dict_dir=self.fit_update_dict_dir,
            n_jobs=self.n_jobs, dict_init=self.dict_init,
            batch_size=self.batch_size, shuffle=self.shuffle,
            verbose=self.verbose, random_state=random_state,
            l1_ratio=self.l1_ratio,
            tol=self.tol,
            return_inner_stats=True,
            return_n_iter=True,
            return_debug_info=self.debug_info)
        self.components_ = U
        # XXX: to remove
        if self.debug_info:
            self.residuals_, self.density_, self.values_ = debug_info
        # Keep track of the state of the algorithm to be able to do
        # some online fitting (partial_fit)
        self.inner_stats_ = (A, B)
        self.iter_offset_ = self.n_iter
        return self

    def partial_fit(self, X, y=None, iter_offset=None):
        """Updates the model using the data in X as a mini-batch.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        iter_offset: integer, optional
            The number of iteration on data batches that has been
            performed before this call to partial_fit. This is optional:
            if no number is passed, the memory of the object is
            used.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        X = check_array(X)
        if hasattr(self, 'components_'):
            dict_init = self.components_
        else:
            dict_init = self.dict_init
        inner_stats = getattr(self, 'inner_stats_', None)
        if iter_offset is None:
            iter_offset = getattr(self, 'iter_offset_', 0)
        U, (A, B) = dict_learning_online(
            X, self.n_components, self.alpha,
            n_iter=self.n_iter, method=self.fit_algorithm,
            update_dict_dir=self.fit_update_dict_dir,
            l1_ratio=self.l1_ratio,
            n_jobs=self.n_jobs, dict_init=dict_init,
            batch_size=len(X), shuffle=False,
            verbose=self.verbose, return_code=False,
            iter_offset=iter_offset, random_state=self.random_state_,
            return_inner_stats=True, inner_stats=inner_stats)
        self.components_ = U

        # Keep track of the state of the algorithm to be able to do
        # some online fitting (partial_fit)
        self.inner_stats_ = (A, B)
        self.iter_offset_ = iter_offset + self.n_iter
        return self
