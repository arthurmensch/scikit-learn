""" Dictionary learning
"""
from __future__ import print_function, division
# Author: Vlad Niculae, Gael Varoquaux, Alexandre Gramfort
# License: BSD 3 clause

import time
import sys
import itertools
from math import sqrt, ceil
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from numpy.lib.stride_tricks import as_strided
from ..base import BaseEstimator, TransformerMixin
from ..externals.joblib import Parallel, delayed, cpu_count
from ..externals.six.moves import zip
from ..utils import (check_array, check_random_state, gen_even_slices,
                     gen_batches, _get_n_jobs, gen_cycling_subsets)
from ..utils.extmath import randomized_svd, row_norms, safe_sparse_dot
from ..utils.validation import check_is_fitted
from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars, \
    ridge_regression
from ..utils.enet_projection import enet_projection, enet_scale, enet_norm


def _sparse_encode(X, dictionary, gram, cov=None, algorithm='lasso_lars',
                   missing_values=None,
                   regularization=None, copy_cov=True,
                   init=None, max_iter=1000, check_input=True, verbose=0,
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

    algorithm: {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than regularization
        from the projection dictionary * data'

    regularization : int | float
        The regularization parameter. It corresponds to alpha when
        algorithm is 'lasso_lars', 'lasso_cd', 'threshold' or 'ridge'
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
    code: array of shape (n_components, n_samples)
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
    n_components = dictionary.shape[0]

    if missing_values is not None and algorithm not in ('lasso_cd', 'ridge') \
            or missing_values not in [0, None]:
        raise NotImplementedError

    if cov is None and algorithm != 'lasso_cd' and missing_values is None:
        # overwriting cov is safe
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    if algorithm == 'lasso_lars':
        # Lars solves (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        alpha = float(regularization) / n_features  # account for scaling
        try:
            err_mgt = np.seterr(all='ignore')
            lasso_lars = LassoLars(alpha=alpha, fit_intercept=False,
                                   verbose=verbose, normalize=False,
                                   precompute=gram, fit_path=False)
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lasso_lars.coef_
        finally:
            np.seterr(**err_mgt)
    elif algorithm == 'lasso_cd':
        # Lasso solves (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        alpha = float(regularization) / n_features  # account for scaling

        # TODO: Make verbosity argument for Lasso?
        # sklearn.linear_model.coordinate_descent.enet_path has a verbosity
        # argument that we could pass in from Lasso.
        clf = Lasso(alpha=alpha, fit_intercept=False, normalize=False,
                    precompute=False if gram is None else gram,
                    max_iter=max_iter, warm_start=True,
                    random_state=True)
        clf.coef_ = init
        if missing_values is None:
            clf.fit(dictionary.T, X.T, check_input=check_input)
            new_code = clf.coef_
        else:
            new_code = np.empty((n_samples, n_components))
            for i in range(n_samples):
                if missing_values == 0:
                    idx = X.indices[X.indptr[i]:X.indptr[i + 1]]
                else:
                    raise NotImplementedError
                if len(idx) != 0:
                    clf.set_params(alpha=clf.alpha * len(idx) / n_features)
                    clf.fit(dictionary.T[idx],
                            X.data[X.indptr[i]:X.indptr[i + 1]])
                    new_code[i] = clf.coef_
    elif algorithm == 'lars':
        try:
            err_mgt = np.seterr(all='ignore')

            # Not passing in verbose=max(0, verbose-1) because Lars.fit already
            # corrects the verbosity level.
            lars = Lars(fit_intercept=False, verbose=verbose, normalize=False,
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
        # TODO: Should verbose argument be passed to this?
        if missing_values is None:
            new_code = orthogonal_mp_gram(
                Gram=gram, Xy=cov, n_nonzero_coefs=int(regularization),
                tol=None, norms_squared=row_norms(X, squared=True),
                copy_Xy=copy_cov).T
        else:
            raise NotImplementedError
    elif algorithm == 'ridge':
        # Ridge solves ||X - DC||^2_2 + alpha * ||C||_2^2,
        # we want to solve 1 / 2 ||X - DC||^2_2 + alpha * ||C||_2^2
        alpha = float(regularization) * 2
        if missing_values is None:
            new_code = ridge_regression(dictionary.T, X.T, alpha,
                                        solver='cholesky')
        else:
            new_code = np.zeros((n_samples, n_components))
            for i in range(n_samples):
                if missing_values == 0:
                    idx = X.indices[X.indptr[i]:X.indptr[i + 1]]
                else:
                    raise NotImplementedError
                if len(idx) != 0:
                    new_code[i] = ridge_regression(dictionary.T[idx],
                                                   X.data[X.indptr[i]:
                                                   X.indptr[i + 1]],
                                                   alpha=alpha * len(idx) /
                                                         n_features,
                                                   solver='cholesky')
    else:
        raise ValueError('Sparse coding method must be "lasso_lars" '
                         '"lasso_cd",  "lasso", "threshold" or "omp",'
                         ' got %s.'
                         % algorithm)
    return new_code


# XXX : could be moved to the linear_model module
def sparse_encode(X, dictionary, missing_values=None, gram=None, cov=None,
                  algorithm='lasso_lars',
                  n_nonzero_coefs=None, alpha=None, copy_cov=True, init=None,
                  max_iter=1000, n_jobs=1, check_input=True, verbose=0,
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

    algorithm: {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection dictionary * X'

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

    check_input: boolean, optional
        If False, the input arrays X and dictionary will not be checked.

    verbose : int, optional
        Controls the verbosity; the higher, the more messages. Defaults to 0.

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
    if check_input:
        if algorithm == 'lasso_cd':
            dictionary = check_array(dictionary, order='C', dtype='float64')
            X = check_array(X, order='C', dtype='float64', accept_sparse='csr')
        elif algorithm == 'ridge':
            dictionary = check_array(dictionary)
            X = check_array(X, accept_sparse='csr')
        else:
            dictionary = check_array(dictionary)
            X = check_array(X)

    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]

    if gram is None and algorithm != 'threshold' and missing_values is None:
        gram = np.dot(dictionary, dictionary.T)

    if cov is None and algorithm != 'lasso_cd' and missing_values is None:
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

    if n_jobs == 1 or algorithm == 'threshold':
        code = _sparse_encode(X, dictionary, gram, cov=cov,
                              algorithm=algorithm,
                              missing_values=missing_values,
                              regularization=regularization, copy_cov=copy_cov,
                              init=init,
                              max_iter=max_iter,
                              check_input=False,
                              random_state=random_state,
                              verbose=verbose,
                              )
        # This ensure that dimensionality of code is always 2,
        # consistant with the case n_jobs > 1
        if code.ndim == 1:
            code = code[np.newaxis, :]
        return code

    # Enter parallel code block
    code = np.empty((n_samples, n_components))

    slices = list(gen_even_slices(n_samples, _get_n_jobs(n_jobs)))
    code_views = Parallel(n_jobs=n_jobs,
                          backend='threading' if
                          algorithm == 'lasso_cd' else 'multiprocessing')(
        delayed(_sparse_encode)(
            X[this_slice], dictionary, gram,
            cov=cov[:, this_slice] if cov is not None else None,
            algorithm=algorithm,
            missing_values=missing_values,
            regularization=regularization, copy_cov=copy_cov,
            init=init[this_slice] if init is not None else None,
            max_iter=max_iter,
            check_input=False,
            random_state=random_state)
        for this_slice in slices)
    for this_slice, this_view in zip(slices, code_views):
        code[this_slice] = this_view

    return code


# def _simpler_update_dict(dictionary, B, A, subset,
#                          seen=None,
#                          return_r2=False,
#                          l1_ratio=0.,
#                          update_support=False,
#                          verbose=False,
#                          shuffle=False,
#                          random_state=None):
#     threshold = 1e-20
#     n_components = len(A)
#     n_features = B.shape[0]
#     random_state = check_random_state(random_state)
#
#     if shuffle:
#         component_range = random_state.permutation(n_components)
#     else:
#         component_range = np.arange(n_components)
#
#     for k in component_range:
#         scale = A[k, k]
#         if update_support:
#             support = np.where(dictionary[:, k] != 0)[0]
#             if seen is not None:
#                 support = np.intersect1d(support, seen, assume_unique=True)
#             this_subset = np.union1d(subset, support)
#         else:
#             this_subset = subset
#         radius = enet_norm(dictionary[this_subset, k], l1_ratio=l1_ratio)
#
#         if radius == 0:
#             radius = 1
#
#         if scale < threshold:
#             dictionary[this_subset, k] = 0
#         else:
#             grad = - B[this_subset, k] + dictionary[this_subset].dot(A[:, k])
#             dictionary[this_subset, k] -= grad / scale
#
#         atom_norm_square = np.sum(dictionary[this_subset, k] ** 2) / radius
#         if atom_norm_square == 0:
#             atom_norm_square = 1
#         # if atom_norm_square < threshold:
#         #     if verbose == 1:
#         #         sys.stdout.write("+")
#         #         sys.stdout.flush()
#         #     elif verbose:
#         #         print("Adding new random atom")
#         #     dictionary[this_subset, k] = random_state.randn(n_features)
#         #     if l1_ratio != 0.:
#         #         # Normalizating new random atom before enet projection
#         #         dictionary[this_subset, k] /= sqrt(atom_norm_square) / radius
#         #     atom_norm_square = np.sum(dictionary[this_subset, k] ** 2)
#         #     # Setting corresponding coefs to 0
#         #     A[k, :] = 0.0
#         #     A[:, k] = 0.0
#         # Projecting onto the norm ball
#         if l1_ratio != 0.:
#             dictionary[this_subset, k] = enet_projection(
#                 dictionary[this_subset, k],
#                 radius=radius,
#                 l1_ratio=l1_ratio,
#                 check_input=True)
#         else:
#             dictionary[this_subset, k] /= sqrt(atom_norm_square)
#     if return_r2:
#         return dictionary, 0
#     else:
#         return dictionary


def _update_dict(dictionary, Y, code,
                 verbose=False,
                 return_r2=False,
                 l1_ratio=0., online=False, full_update=True,
                 shuffle=False,
                 random_state=None):
    """Update the dense dictionary factor in place, constraining dictionary
    component to have a unit l2 norm.

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
        Whether the update we perform is part of an online algorithm or not
        (this changes derivation of residuals and of step size).

    shuffle: bool,
        Whether to shuffle the components when performing sequential
        coordinate update.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    dictionary: array of shape (n_features, n_components)
        Updated dictionary.

    """
    threshold = 1e-20

    n_components = len(code)
    n_features = Y.shape[0]
    random_state = check_random_state(random_state)

    radius = enet_norm(dictionary.T, l1_ratio=l1_ratio)
    radius[radius == 0] = 1

    if full_update:
        restart_radius = 1

    # Residuals, computed 'in-place' for efficiency
    R = -np.dot(code.T, dictionary.T).T
    R += Y
    R = np.asfortranarray(R)
    ger, = linalg.get_blas_funcs(('ger',), (dictionary, code))

    if shuffle:
        component_range = random_state.permutation(n_components)
    else:
        component_range = np.arange(n_components)

    for k in component_range:
        # R <- 1.0 * U_k * V_k^T + R
        R = ger(1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
        # Coordinate update
        if online:
            scale = code[k, k]
        else:
            scale = np.sum(code[k, :] ** 2)
        if scale < threshold:
            # Trigger cleaning
            dictionary[:, k] = 0
        else:
            if online:
                dictionary[:, k] = R[:, k] / scale
            else:
                dictionary[:, k] = np.dot(R, code[k, :].T) / scale

        new_radius = enet_norm(dictionary[:, k], l1_ratio=l1_ratio)

        if new_radius < threshold:
            # Cleaning small atoms
            if full_update:
                if verbose == 1:
                    sys.stdout.write("+")
                    sys.stdout.flush()
                elif verbose:
                    print("Adding new random atom")

                dictionary[:, k] = enet_scale(random_state.randn(n_features),
                                              l1_ratio=l1_ratio,
                                              radius=restart_radius)

                # Setting corresponding coefs to 0
                code[k, :] = 0.0
                if online:
                    code[:, k] = 0
            else:
                dictionary[:, k] = 0
        else:
            # Projecting onto the norm ball
            dictionary[:, k] = enet_projection(dictionary[:, k],
                                               radius=radius[k],
                                               l1_ratio=l1_ratio,
                                               check_input=False)
        # R <- -1.0 * U_k * V_k^T + R
        R = ger(-1.0, dictionary[:, k], code[k, :],
                a=R, overwrite_a=True)

    if return_r2:
        if online:
            # Y = B_t, code = A_t, dictionary = D in online setting
            R += Y
            # residual = 1 / 2 Tr(D^T D A_t) - Tr(D^T B_t)
            residual = -np.sum(dictionary * R) / 2
        else:
            R **= 2
            # R is fortran-ordered. For numpy version < 1.6, sum does not
            # follow the quick striding first, and is thus inefficient on
            # fortran ordered data. We take a flat view of the data with no
            # striding
            R = as_strided(R, shape=(R.size,), strides=(R.dtype.itemsize,))
            residual = np.sum(R)

    if return_r2:
        return dictionary, residual
    else:
        return dictionary


def dict_learning(X, n_components, alpha, l1_ratio=0, max_iter=100, tol=1e-8,
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
    if method not in ('lars', 'cd', 'ridge'):
        raise ValueError('Coding method not supported as a fit algorithm.')
    if method in ('lars', 'cd'):
        method = 'lasso_' + method

    t0 = time.time()
    n_samples, n_features = X.shape

    l1_ratio = float(l1_ratio)

    t0 = time.time()
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    X = check_array(X, dtype=np.float64)

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

    dictionary = np.array(dictionary, order='C', dtype='float64', copy=False)

    residuals = 0

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print('[dict_learning]', end=' ')

    radius = 1
    dictionary = enet_scale(dictionary, l1_ratio=l1_ratio,
                            radius=radius, inplace=True)
    # If max_iter is 0, number of iterations returned should be zero
    ii = -1

    for ii in range(max_iter):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print("Iteration % 3i "
                  "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                  % (ii, dt, dt / 60, current_cost))

        # Update code
        code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha,
                             init=code, n_jobs=n_jobs, check_input=False,
                             random_state=random_state)
        # Update dictionary
        dictionary, residuals = _update_dict(dictionary.T, X.T, code.T,
                                             verbose=verbose, return_r2=True,
                                             online=False,
                                             shuffle=False,
                                             random_state=random_state,
                                             full_update=True,
                                             l1_ratio=l1_ratio)
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


def dict_learning_online(X, n_components=2, alpha=1,
                         l1_ratio=0.0, n_iter=100,
                         return_code=True, dict_init=None,
                         batch_size=3, verbose=False, shuffle=True, n_jobs=1,
                         method='lars',
                         iter_offset=0, tol=0.,
                         learning_rate=.5,
                         mask_subsets=None,
                         feature_ratio=1,
                         missing_values=None,
                         random_state=None,
                         return_inner_stats=False, inner_stats=None,
                         return_n_iter=False,
                         return_debug_info=False):
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
        Sparsity controlling parameter if `method='lars'` or `method='cd'
        Regularization parameter if `method='ridge'` : increasing it will also
        increase dictionary regularity and sparsity.

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
        ridge: compute code using an ordinary least square method.

    l1_ratio: float,
        Sparsity controlling parameter for dictionary projection.
        The higher it is, the sparser the dictionary component will be.

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
        Whether to keep track of objective value, sparsity value and to record
        up to of 100 dictionary trajectory

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
    :param update_scheme:

    """
    if n_components is None:
        n_components = X.shape[1]

    if missing_values not in [None, 0]:
        raise ValueError('Missing value should be 0 or None')
    if missing_values == 0 and not sp.isspmatrix_csr(X):
        raise ValueError('X should be provided in csr format when using missing'
                         ' values')

    if method not in ('lars', 'cd', 'ridge'):
        raise ValueError('Coding method not supported as a fit algorithm.')
    if method in ('lars', 'cd'):
        method = 'lasso_' + method

    t0 = time.time()
    n_samples, n_features = X.shape

    l1_ratio = float(l1_ratio)

    random_state = check_random_state(random_state)

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Init V with SVD of X
    if dict_init is not None:
        dictionary = dict_init
    else:
        _, S, dictionary = randomized_svd(X, n_components, n_iter=5,
                                          random_state=random_state)
        dictionary = S[:, np.newaxis] * dictionary
    r = dictionary.shape[0]
    if n_components <= r:
        dictionary = dictionary[:n_components, :]
    else:
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    if verbose == 1:
        print('[dict_learning]', end=' ')

    dictionary = check_array(dictionary.T, order='F', dtype=np.float64,
                             copy=False, accept_sparse='csc')
    X = check_array(X, accept_sparse='csr', order='C', dtype=np.float64,
                    copy=False)


    batches = gen_batches(n_samples, batch_size)
    batches = itertools.cycle(batches)


    if missing_values not in [0, None]:
        raise NotImplementedError

    if feature_ratio == 1.:
        mask_subsets = itertools.repeat(slice(None))
    elif mask_subsets is None:
        mask_subsets = gen_cycling_subsets(n_features,
                                           n_features / feature_ratio,
                                           random=(feature_ratio > 1),
                                           random_state=random_state)

    radius = 1  # sqrt(n_features)
    if n_iter != 0 and iter_offset == 0 and inner_stats is None:
        enet_scale(dictionary.T, l1_ratio=l1_ratio,
                   radius=radius,
                   inplace=True)
    if inner_stats is None:
        # The covariance of the dictionary
        A = np.zeros((n_components, n_components))
        # The data approximation
        B = np.zeros_like(dictionary)

        # Ap = np.zeros((n_components, n_components))
        # Bp = np.zeros_like(dictionary)

        last_cost = np.inf
        norm_cost = 0
        penalty_cost = 0
        n_seen_samples = 0
        count_seen_features = np.zeros(n_features)
        A_ref = np.zeros((n_components, n_components))
        B_ref = np.zeros_like(dictionary)
    else:
        A = inner_stats[0].copy()
        B = inner_stats[1].copy()

        # Ap = inner_stats[2].copy()
        # Bp = inner_stats[3].copy()

        last_cost = inner_stats[2][0]
        norm_cost = inner_stats[2][1]
        penalty_cost = inner_stats[2][2]
        n_seen_samples = inner_stats[2][3]
        count_seen_features = inner_stats[2][4]
        A_ref = inner_stats[2][5].copy()
        B_ref = inner_stats[2][6].copy()

    if return_debug_info:
        debug_info = {'density': [],
                      'values': [],
                      'residuals': [],
                      'norm_cost': [],
                      'penalty_cost': [],
                      'objective_cost': []}
        size_values = min(n_features, 100)
        # recorded_features = np.floor(np.linspace(0, n_features - 1,
        #                                          size_values)).astype('int')
        recorded_features = np.arange(size_values)

    # For tolerance computation
    patience = 0

    # If n_iter is zero, we need to return zero.
    ii = iter_offset - 1

    if shuffle:
        permutation = random_state.permutation(n_samples)

    total_time = 0

    for ii, batch, mask_subset in zip(range(iter_offset, iter_offset + n_iter),
                                      batches, mask_subsets):

        t1 = time.time()

        if shuffle:
            this_X = X[permutation[batch]]
        else:
            this_X = X[batch]

        if missing_values is None:
            existing = slice(None)
        else:
            existing = np.array([], dtype='int')
            if sp.isspmatrix_csr(this_X):
                for i in range(this_X.shape[0]):
                    sample_existing = this_X.indices[
                                      this_X.indptr[i]:this_X.indptr[i + 1]]
                    existing = np.union1d(sample_existing, existing)
            else:
                existing = np.unique(np.nonzero(this_X)[1])
            if len(existing) == 0:
                # No samples : skip
                continue

        if isinstance(mask_subset, slice) and mask_subset == slice(None):
            subset = existing
        elif isinstance(existing, slice) and existing == slice(None):
            subset = mask_subset
        else:
            subset = np.intersect1d(mask_subset, existing)
        print(subset)

        full_update = (isinstance(subset, slice) and subset == slice(None))

        if full_update:
            ratio = 1
            subset_dictionary = dictionary
        else:
            ratio = len(subset) / n_features
            subset_dictionary = check_array(dictionary[subset], order='F')

        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            if verbose > 10 or ii % ceil(100. / verbose) == 0:
                print("Iteration % 3i (elapsed time: % 3is, % 4.1fmn)"
                      % (ii, dt, dt / 60))

        len_batch = batch.stop - batch.start
        n_seen_samples += len_batch
        count_seen_features[subset] += len_batch

        this_code = sparse_encode(
            this_X[:, subset],
            subset_dictionary.T,
            algorithm=method,
            alpha=alpha * ratio,
            n_jobs=1,
            check_input=False,
            missing_values=missing_values,
            random_state=random_state).T

        A *= 1 - len_batch / pow(n_seen_samples, learning_rate)
        A += np.dot(this_code, this_code.T) / pow(n_seen_samples,
                                                  learning_rate)
        B[subset] *= 1 - len_batch / np.power(count_seen_features[subset,
                                                                 np.newaxis],
                                              learning_rate)
        B[subset] += safe_sparse_dot(this_X[:, subset].T,
                                     this_code.T) / np.power(
            count_seen_features[subset, np.newaxis], learning_rate)

        # Update dictionary
        dictionary[subset] = _update_dict(
            subset_dictionary,
            B[subset], A,
            verbose=verbose,
            l1_ratio=l1_ratio,
            random_state=random_state,
            return_r2=False,
            online=True,
            full_update=full_update,
            shuffle=shuffle)
        total_time += time.time() - t1

        A_ref *= (1 - len_batch / pow(n_seen_samples, learning_rate))
        A_ref += np.dot(this_code, this_code.T) / pow(n_seen_samples,
                                                      learning_rate)
        B_ref *= (1 - len_batch / pow(n_seen_samples, learning_rate))
        B_ref += safe_sparse_dot(this_X.T, this_code.T) / pow(n_seen_samples,
                                                              learning_rate)
        total_time += time.time() - t0
        objective_cost = .5 * np.sum(dictionary.T.dot(dictionary) * A_ref)
        objective_cost -= np.sum(dictionary * B_ref)
        # Residual computation
        norm_cost *= (1 - len_batch / pow(n_seen_samples, learning_rate))
        if not full_update:
            if missing_values is not None:
                norm_cost += .5 * np.sum(this_X.data ** 2) * n_features / len(
                    subset) / pow(n_seen_samples, learning_rate)
            else:
                norm_cost += .5 * np.sum(this_X ** 2)\
                             / pow(n_seen_samples, learning_rate)
        else:
            norm_cost += .5 * np.sum(this_X ** 2) / pow(n_seen_samples,
                                                        learning_rate)

        penalty_cost *= (1 - len_batch / pow(n_seen_samples, learning_rate))
        if method in ('lasso_lars', 'lasso_cd'):
            penalty_cost += alpha * np.sum(
                np.abs(this_code)) / pow(n_seen_samples, learning_rate)
        else:
            penalty_cost += alpha * np.sum(this_code ** 2) / pow(n_seen_samples, learning_rate)
        current_cost = objective_cost + norm_cost + penalty_cost

        # Stopping criterion
        if abs(last_cost - current_cost) < tol * current_cost:
            patience += 1
        else:
            patience = 0
        last_cost = current_cost

        # XXX to remove
        if return_debug_info:
            debug_info['values'].append((dictionary[:, 0][recorded_features]))
            debug_info['density'].append(
                1 - float(np.sum(dictionary == 0.)) / np.size(dictionary))
            debug_info['residuals'].append(current_cost)
            debug_info['norm_cost'].append(norm_cost)
            debug_info['penalty_cost'].append(penalty_cost)
            debug_info['objective_cost'].append(objective_cost)

        if patience >= 3:
            if verbose == 1:
                # A line return
                print("")
            elif verbose:
                print("--- Convergence reached after %d iterations" % ii)
            break

    residual_stat = (last_cost, norm_cost, penalty_cost, n_seen_samples,
                     count_seen_features, A_ref, B_ref)
    if return_debug_info:
        debug_info['total_time'] = total_time
        debug_info['count_seen_features'] = count_seen_features

    if return_inner_stats:
        if return_n_iter:
            res = dictionary.T, (
                A, B, residual_stat), ii - iter_offset + 1
        else:
            res = dictionary.T, (A, B, residual_stat)
    elif return_code:
        if verbose > 1:
            print('Learning code...', end=' ')
        elif verbose == 1:
            print('|', end=' ')
        code = sparse_encode(X, dictionary.T, algorithm=method, alpha=alpha,
                             missing_values=missing_values,
                             n_jobs=n_jobs, check_input=False)
        if verbose > 1:
            dt = (time.time() - t0)
            print('done (total time: % 3is, % 4.1fmn)' % (dt, dt / 60))
        if return_n_iter:
            res = code, dictionary.T, ii - iter_offset + 1
        else:
            res = code, dictionary.T

    elif return_n_iter:
        res = dictionary.T, ii - iter_offset + 1
    else:
        res = dictionary.T

    if return_debug_info:
        return res, debug_info
    else:
        return res


class SparseCodingMixin(TransformerMixin):
    """Sparse coding mixin"""

    def _set_sparse_coding_params(self, n_components,
                                  transform_algorithm='omp',
                                  transform_n_nonzero_coefs=None,
                                  transform_alpha=None, split_sign=False,
                                  missing_values=None,
                                  n_jobs=1):
        self.n_components = n_components
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.split_sign = split_sign
        self.n_jobs = n_jobs
        self.missing_values = missing_values

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
        X = check_array(X, accept_sparse='csr')
        n_samples, n_features = X.shape

        code = sparse_encode(
            X, self.components_, algorithm=self.transform_algorithm,
            missing_values=self.missing_values,
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
    'threshold', 'ridge'}
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
                 split_sign=False, missing_values=None, n_jobs=1):
        self._set_sparse_coding_params(dictionary.shape[0],
                                       transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign,
                                       missing_values, n_jobs)
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

    def __init__(self, n_components=None, alpha=1, l1_ratio=0.0,
                 max_iter=1000, tol=1e-8,
                 fit_algorithm='lars', transform_algorithm='omp',
                 transform_n_nonzero_coefs=None, transform_alpha=None,
                 missing_values=None,
                 n_jobs=1, code_init=None, dict_init=None, verbose=False,
                 split_sign=False, random_state=None):

        self._set_sparse_coding_params(n_components, transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign,
                                       missing_values, n_jobs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
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
            l1_ratio=self.l1_ratio,
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
        sparsity controlling parameter

    l1_ratio: float,
        sparsity controlling parameter for dictionary component

    tol: float,
        Tolerance for the stopping condition. 0. to disable

    n_iter : int,
        total number of iterations to perform

    fit_algorithm : {'lars', 'cd', 'ridge'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.
        ridge: use a ridge penalty on U : alpha * || U ||_2^2, yielding non
        sparse code

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
        ridge: uses a penalized least square fit

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

    inner_stats_ : tuple of (A, B, residuals_stat) ndarrays
        Internal sufficient statistics that are kept by the algorithm.
        Keeping them is useful in online settings, to avoid loosing the
        history of the evolution, but they shouldn't have any use for the
        end user.
        A (n_components, n_components) is the dictionary covariance matrix.
        B (n_features, n_components) is the data approximation matrix
        residuals_stat tuple of (residuals, residuals_penalty,
        residuals_normalization) keeps values necessary for residual
        computation and convergence analysis

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

    def __init__(self, n_components=None, alpha=1, dict_penalty=0,
                 learning_rate=1,
                 l1_ratio=0.0,
                 n_iter=1000, fit_algorithm='lars', n_jobs=1,
                 batch_size=3, tol=0., shuffle=True, dict_init=None,
                 transform_algorithm='omp',
                 missing_values=None,
                 transform_n_nonzero_coefs=None, transform_alpha=None,
                 verbose=False, split_sign=False,
                 random_state=None,
                 debug_info=False,
                 feature_ratio=1):
        self._set_sparse_coding_params(n_components,
                                       transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign,
                                       missing_values, n_jobs)
        self.alpha = alpha
        self.dict_penalty = dict_penalty
        self.l1_ratio = l1_ratio
        self.n_iter = n_iter
        self.fit_algorithm = fit_algorithm
        self.dict_init = dict_init
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.split_sign = split_sign
        self.random_state = random_state
        self.tol = tol
        self.debug_info = debug_info
        self.learning_rate = learning_rate
        self.feature_ratio = feature_ratio

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the   number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.random_state_ = check_random_state(self.random_state)
        X = check_array(X, accept_sparse='csr')

        if self.feature_ratio == 1:
            self.mask_subsets_ = itertools.repeat(None)
        else:
            self.mask_subsets_ = gen_cycling_subsets(X.shape[1],
                                                     batch_size=int(
                X.shape[1] / self.feature_ratio),
                                                     random=self.feature_ratio > 1,
                                                     random_state=self.random_state_)

        res = dict_learning_online(
            X, self.n_components, self.alpha,
            l1_ratio=self.l1_ratio,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter, return_code=False,
            method=self.fit_algorithm,
            missing_values=self.missing_values,
            n_jobs=self.n_jobs, dict_init=self.dict_init,
            batch_size=self.batch_size, shuffle=self.shuffle,
            verbose=self.verbose, random_state=self.random_state_,
            tol=self.tol,
            # To be able to run partial_fit behind a fit
            return_inner_stats=True,
            return_n_iter=True,
            return_debug_info=self.debug_info,
            feature_ratio=self.feature_ratio,
            mask_subsets=self.mask_subsets_)

        if self.debug_info:
            (U, self.inner_stats_, n_iter), debug_info = res
            if not hasattr(self, 'debug_info_'):
                self.debug_info_ = debug_info
            else:
                for key in self.debug_info_:
                    self.debug_info_[key] += debug_info[key]
        else:
            U, self.inner_stats_, n_iter = res

        self.n_iter_ = n_iter
        self.components_ = U
        return self

    def partial_fit(self, X, y=None, iter_offset=None, deprecated=True):
        """Updates the model using the data in X

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
        X = check_array(X, accept_sparse='csr')
        if not hasattr(self, 'mask_subsets_'):
            if self.feature_ratio == 1:
                self.mask_subsets_ = itertools.repeat(None)
            else:
                self.mask_subsets_ = gen_cycling_subsets(
                    X.shape[1], batch_size=int(X.shape[1] / self.feature_ratio),
                    random_state=self.random_state_)
        if hasattr(self, 'components_'):
            dict_init = self.components_
        else:
            dict_init = self.dict_init
        inner_stats = getattr(self, 'inner_stats_', None)
        if iter_offset is None:
            iter_offset = getattr(self, 'n_iter_', 0)

        if not deprecated:
            # Doing one pass on the data, ignoring self.n_iter
            n_iter = (X.shape[0] - 1) // self.batch_size + 1
            batch_size = self.batch_size
        else:
            n_iter = self.n_iter
            batch_size = X.shape[0]
        res = dict_learning_online(
            X, self.n_components, self.alpha,
            l1_ratio=self.l1_ratio,
            n_iter=n_iter,
            tol=0,
            missing_values=self.missing_values,
            feature_ratio=self.feature_ratio,
            mask_subsets=self.mask_subsets_,
            method=self.fit_algorithm,
            n_jobs=self.n_jobs, dict_init=dict_init,
            batch_size=batch_size,
            shuffle=self.shuffle,
            verbose=self.verbose, return_code=False,
            iter_offset=iter_offset, random_state=self.random_state_,
            return_inner_stats=True, inner_stats=inner_stats,
            return_debug_info=self.debug_info)

        if self.debug_info:
            (U, self.inner_stats_), debug_info = res
            if not hasattr(self, 'debug_info_'):
                self.debug_info_ = debug_info
            else:
                for key in self.debug_info_:
                    self.debug_info_[key] += debug_info[key]
        else:
            U, self.inner_stats_ = res

        self.n_iter_ = iter_offset + n_iter
        self.components_ = U
        return self
