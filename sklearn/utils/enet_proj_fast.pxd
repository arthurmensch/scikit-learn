"""Projection on the elastic-net ball (Cython version)
**References:**

J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
for sparse coding (http://www.di.ens.fr/sierra/pdfs/icml09.pdf)
"""

# Author: Arthur Mensch
# License: BSD

cimport cython
cimport numpy as np

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t
ctypedef np.uint8_t UINT8_t
cdef enum:
    MASKED = 0
    UNMASKED = 1
    GREATER = 2
    LOWER = 3
cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

cdef double _enet_projection_with_mask(DOUBLE[:] res, DOUBLE[:] v, UINT8_t * mask, double radius,
                             double l1_ratio, UINT32_t random_state) nogil