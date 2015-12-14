"""Matrix factorization with Sparse PCA"""
# Author: Vlad Niculae, Gael Varoquaux, Alexandre Gramfort
# License: BSD 3 clause

import numpy as np
from ..utils import check_random_state, check_array
from ..utils.validation import check_is_fitted
from ..linear_model import ridge_regression
from ..base import BaseEstimator, TransformerMixin
from .dict_learning import dict_learning, dict_learning_online, \
    MiniBatchDictionaryLearning


class SparsePCA(BaseEstimator, TransformerMixin):
    """Sparse Principal Components Analysis (SparsePCA)

    Finds the set of sparse components that can optimally reconstruct
    the data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha.

    Read more in the :ref:`User Guide <SparsePCA>`.

    Parameters
    ----------
    n_components : int,
        Number of sparse atoms to extract.

    alpha : float,
        Sparsity controlling parameter. Higher values lead to sparser
        components.

    ridge_alpha : float,
        Amount of ridge shrinkage to apply in order to improve
        conditioning when calling the transform method.

    max_iter : int,
        Maximum number of iterations to perform.

    tol : float,
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs : int,
        Number of parallel jobs to run.

    U_init : array of shape (n_samples, n_components),
        Initial values for the loadings for warm restart scenarios.

    V_init : array of shape (n_components, n_features),
        Initial values for the components for warm restart scenarios.

    verbose :
        Degree of verbosity of the printed output.

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Sparse components extracted from the data.

    error_ : array
        Vector of errors at each iteration.

    n_iter_ : int
        Number of iterations run.

    See also
    --------
    PCA
    MiniBatchSparsePCA
    DictionaryLearning
    """

    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, tol=1e-8, method='lars', n_jobs=1, U_init=None,
                 V_init=None, verbose=False, random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_jobs = n_jobs
        self.U_init = U_init
        self.V_init = V_init
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
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components
        code_init = self.V_init.T if self.V_init is not None else None
        dict_init = self.U_init.T if self.U_init is not None else None
        Vt, _, E, self.n_iter_ = dict_learning(X.T, n_components, self.alpha,
                                               tol=self.tol,
                                               max_iter=self.max_iter,
                                               method=self.method,
                                               n_jobs=self.n_jobs,
                                               verbose=self.verbose,
                                               random_state=random_state,
                                               code_init=code_init,
                                               dict_init=dict_init,
                                               return_n_iter=True
                                               )
        self.components_ = Vt.T
        self.error_ = E
        return self

    def transform(self, X, ridge_alpha=None):
        """Least Squares projection of the data onto the sparse components.

        To avoid instability issues in case the system is under-determined,
        regularization can be applied (Ridge regression) via the
        `ridge_alpha` parameter.

        Note that Sparse PCA components orthogonality is not enforced as in PCA
        hence one cannot use a simple linear projection.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        ridge_alpha: float, default: 0.01
            Amount of ridge shrinkage to apply in order to improve
            conditioning.

        Returns
        -------
        X_new array, shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, 'components_')

        X = check_array(X)
        ridge_alpha = self.ridge_alpha if ridge_alpha is None else ridge_alpha
        U = ridge_regression(self.components_.T, X.T, ridge_alpha,
                             solver='cholesky')
        s = np.sqrt((U ** 2).sum(axis=0))
        s[s == 0] = 1
        U /= s
        return U


class MiniBatchSparsePCA(SparsePCA):
    """Mini-batch Sparse Principal Components Analysis

    Finds the set of sparse components that can optimally reconstruct
    the data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha.

    Read more in the :ref:`User Guide <SparsePCA>`.

    Parameters
    ----------
    n_components : int,
        number of sparse atoms to extract

    alpha : int,
        Sparsity controlling parameter. Higher values lead to sparser
        components.

    ridge_alpha : float,
        Amount of ridge shrinkage to apply in order to improve
        conditioning when calling the transform method.

    n_iter : int,
        number of iterations to perform for each mini batch

    callback : callable,
        callable that gets invoked every five iterations

    batch_size : int,
        the number of features to take in each mini batch

    verbose :
        degree of output the procedure will print

    shuffle : boolean,
        whether to shuffle the data before splitting it in batches

    n_jobs : int,
        number of parallel jobs to run, or -1 to autodetect.

    method : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Sparse components extracted from the data.

    error_ : array
        Vector of errors at each iteration.

    n_iter_ : int
        Number of iterations run.

    See also
    --------
    PCA
    SparsePCA
    DictionaryLearning
    """

    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01,
                 n_iter=100, callback=None, batch_size=3, verbose=False,
                 shuffle=True, n_jobs=1, method='lars', random_state=None,
                 feature_ratio=1):

        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.n_iter = n_iter
        self.callback = callback
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.method = method
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
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components
        Vt, _, self.n_iter_ = dict_learning_online(
            X.T, n_components, alpha=self.alpha,
            n_iter=self.n_iter, return_code=True,
            dict_init=None, verbose=self.verbose,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            n_jobs=self.n_jobs, method=self.method,
            random_state=random_state,
            return_n_iter=True)
        self.components_ = Vt.T
        return self


class IncrementalSparsePCA(MiniBatchDictionaryLearning):
    def __init__(self, n_components=None, alpha=1, l1_ratio=1.0,
                 dict_penalty=0,
                 n_iter=1000, n_jobs=1,
                 batch_size=3, tol=0., shuffle=True, dict_init=None,
                 transform_alpha=None,
                 missing_values=None,
                 learning_rate=1,
                 verbose=False, random_state=None,
                 debug_info=False,
                 feature_ratio=1):
        MiniBatchDictionaryLearning.__init__(self, n_components=n_components,
                                             alpha=alpha,
                                             dict_penalty=dict_penalty,
                                             l1_ratio=l1_ratio,
                                             learning_rate=learning_rate,
                                             fit_algorithm='ridge',
                                             transform_algorithm='ridge',
                                             n_iter=n_iter,
                                             n_jobs=n_jobs,
                                             missing_values=missing_values,
                                             batch_size=batch_size,
                                             tol=tol,
                                             shuffle=shuffle,
                                             dict_init=dict_init,
                                             transform_alpha=transform_alpha,
                                             verbose=verbose,
                                             random_state=random_state,
                                             feature_ratio=feature_ratio,
                                             split_sign=False,
                                             debug_info=debug_info)
