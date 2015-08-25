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


def enet_scale(v, l1_ratio=0.1, radius=1):
    l1_dictionary = np.sum(np.abs(v), axis=1) * l1_ratio
    l2_dictionary = np.sum(v ** 2, axis=1) * (1 - l1_ratio)
    S = - l1_dictionary + np.sqrt(l1_dictionary ** 2 + 4 *
                                  radius * l2_dictionary)
    S /= 2 * l2_dictionary
    S[S == 0] = 1
    return v * S[:, np.newaxis]


def enet_threshold(v, l1_ratio=0.1, radius=1):
    Sv = np.sqrt(np.sum(v ** 2, axis=1)) / radius
    Sv[Sv == 0] = 1
    b = enet_projection(v / Sv[:, np.newaxis], l1_ratio=l1_ratio,
                        radius=radius)
    Sb = np.sqrt(np.sum(b ** 2, axis=1)) / radius
    Sb[Sb == 0] = 1
    b /= Sb / Sv