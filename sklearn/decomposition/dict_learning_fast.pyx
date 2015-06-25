"""
Dictionary update loop with elastic net constraint on single feature vector
"""
# Author: Arthur Mensch
# License: BSD 3 clause

cimport cython
cimport numpy as np
from ..utils.enet_proj_fast cimport _enet_projection_with_mask, DOUBLE, UINT8_t, UINT32_t
from ..utils._random cimport sample_without_replacement

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf


cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        AtlasConj=114

    void dgemv "cblas_dgemv"(CBLAS_ORDER Order,
                      CBLAS_TRANSPOSE TransA, int M, int N,
                      double alpha, double *A, int lda,
                      double *X, int incX, double beta,
                      double *Y, int incY) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef void _update_dict_fast(DOUBLE[:, :] dictionary, DOUBLE[:, :] R,
                             DOUBLE[:, :] code,
                             double l1_ratio,
                             double radius) nogil:
    cdef int n_features = dictionary.shape[0]
    cdef int n_components = dictionary.shape[1]
    cdef int n_samples = code.shape[1]
    cdef int j
    cdef int idx
    cdef double atom_norm_square
    cdef double * dictionary_ptr = &dictionary[0, 0]
    cdef double * code_ptr = &code[0,0]
    cdef double * R_ptr = &R[0, 0]
    cdef UINT8_t * mask
    cdef double norm
    cdef int random_state = 0
    with nogil:
        # Initializing mask here to avoid too much malloc
        mask = <UINT8_t *>malloc(n_features * sizeof(UINT8_t))

        for j in range(n_features):
            # R[j, :] += np.dot(dictionary[j, :], code)
            dgemv(CblasRowMajor, CblasTrans, n_components, n_samples, 1., code_ptr, n_components,
                 dictionary_ptr + j * n_components, 1, 1., R_ptr + j * n_samples, 1)
            norm = _enet_projection_with_mask(dictionary[j, :], R[j, :], mask, radius, l1_ratio, random_state)
            dgemv(CblasRowMajor, CblasTrans, n_components, n_samples, -1., code_ptr, n_components,
                  dictionary_ptr + j * n_components, 1, 1., R_ptr + j * n_samples, 1)
        free(mask)
