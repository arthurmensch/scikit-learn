"""
Dictionary update loop with elastic net constraint along features. EXPERIMENTAL
"""
# Author: Arthur Mensch
# License: BSD 3 clause

cimport cython

from ..utils.enet_proj_fast cimport _enet_projection_with_mask, DOUBLE, UINT8_t, UINT32_t, INT64_t

from libc.stdlib cimport malloc, free


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
cpdef void _update_dict_feature_wise_fast(DOUBLE[:, :] dictionary, DOUBLE[:, :] R,
                             DOUBLE[:, :] code,
                             long[:] permutation,
                             double l1_ratio,
                             double radius) nogil:
    cdef int n_features = permutation.shape[0]
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
        mask = <UINT8_t *>malloc(n_components * sizeof(UINT8_t))
        for idx in range(n_features):
            j = permutation[idx]
            # R[j, :] += np.dot(dictionary[j, :], code)
            dgemv(CblasRowMajor, CblasTrans, n_components, n_samples, 1., code_ptr, n_components,
                 dictionary_ptr + j * n_components, 1, 1., R_ptr + j * n_samples, 1)
            norm = _enet_projection_with_mask(dictionary[j, :], R[j, :], mask, radius, l1_ratio, random_state)
            # R[j, :] -= np.dot(dictionary[j, :], code)
            dgemv(CblasRowMajor, CblasTrans, n_components, n_samples, -1., code_ptr, n_components,
                  dictionary_ptr + j * n_components, 1, 1., R_ptr + j * n_samples, 1)
        free(mask)
