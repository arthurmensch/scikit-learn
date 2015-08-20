"""Projection on the elastic-net ball
**References:**

J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
for sparse coding (http://www.di.ens.fr/sierra/pdfs/icml09.pdf)
"""
import numpy as np
from .enet_proj_fast import enet_projection_inplace
from .enet_proj_fast import enet_norm as c_enet_norm
from . import check_array


def enet_projection(v, radius=1., l1_ratio=0.1, check_input=True):
    if check_input:
        v = check_array(v, dtype=np.float64, order='C', copy=False)
    b = np.zeros(v.shape, dtype=np.float64)
    if v.ndim == 1:
        enet_projection_inplace(v, b, radius, l1_ratio)
    else:
        for i in range(v.shape[0]):
            enet_projection_inplace(v[i], b[i], radius, l1_ratio)
    return b


def enet_norm(v, l1_ratio=0.1):
    v = check_array(v, dtype=np.float64, order='C', copy=False)
    if v.ndim == 1:
        return c_enet_norm(v, l1_ratio)
    else:
        m = v.shape[0]
        norms = np.zeros(m, dtype=np.float64)
        for i in range(m):
            norms[i] = c_enet_norm(v[i], l1_ratio)
    return norms