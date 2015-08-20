# Author: Arthur Mensch
# License: BSD 3 clause

import numpy as np
from numpy import sqrt
from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.enet_projection import enet_norm, enet_projection


def enet_norm_slow(v, l1_ratio=0.1):
    b_abs = np.abs(v)
    return np.sum(b_abs * (l1_ratio + (1 - l1_ratio) * b_abs))


def swap(b, i, j):
    if i != j:
        buf = b[i]
        b[i] = b[j]
        b[j] = buf


def enet_projection_slow(v, radius=1, l1_ratio=0.1):
    b = np.abs(v)
    random_state = check_random_state(None)
    if l1_ratio == 0:
        norm = sqrt(np.sum(b ** 2))
        if norm <= radius:
            return b
        else:
            return b / norm * sqrt(radius)
    gamma = 2 / l1_ratio - 2
    radius /= l1_ratio
    m = b.shape[0]
    b_abs = np.abs(b)
    norm = enet_norm_slow(b_abs, l1_ratio)
    if norm <= radius:
        return b
    else:
        s = 0
        rho = 0
        start_U = 0
        size_U = m
        while size_U > 0:
            pivot = random_state.randint(size_U) + start_U
            # Putting pivot at the beginning
            swap(b, pivot, start_U)
            pivot = start_U
            drho = 1
            ds = b[pivot] * (1 + gamma / 2 * b[pivot])
            # Ordering : [pivot, >=, <], using Lobato quicksort
            for i in range(start_U + 1, start_U + size_U):
                if b[i] >= b[pivot]:
                    ds += b[i] * (1 + gamma / 2 * b[i])
                    swap(b, i, start_U + drho)
                    drho += 1
            if s + ds - (rho + drho) * (1 + gamma / 2 * b[pivot])\
                    * b[pivot] < radius * (1 + gamma * b[pivot]) ** 2:
                # U <- L : [<]
                start_U += drho
                size_U -= drho
                rho += drho
                s += ds
            else:
                # U <- G \ k : [>=]
                start_U += 1
                size_U = drho - 1
        if gamma != 0:
            a = gamma ** 2 * radius + gamma * rho * 0.5
            d = 2 * radius * gamma + rho
            c = radius - s
            l = (-d + np.sqrt(d ** 2 - 4 * a * c)) / (2*a)
        else:
            l = (s - radius) / rho
        v_sign = np.sign(v)
        v_sign[v_sign == 0] = 1
        return v_sign * np.maximum(0, np.abs(v) - l) / (1 + l * gamma)


def test_enet_norm_slow():
    norms = np.zeros(10)
    norms2 = np.zeros(10)
    random_state = check_random_state(0)

    for i in range(10):
        a = random_state.randn(10000)
        norms[i] = enet_norm_slow(a, l1_ratio=0.1)
        norms2[i] = 0.9 * (a ** 2).sum() + 0.1 * np.abs(a).sum()
    assert_array_almost_equal(norms, norms2)


def test_enet_projection_slow_norm():
    norms = np.zeros(10)
    random_state = check_random_state(0)

    for i in range(10):
        a = random_state.randn(10000)
        b = np.asarray(enet_projection_slow(a, radius=1, l1_ratio=0.1))
        norms[i] = enet_norm_slow(b, l1_ratio=0.1)
    assert_array_almost_equal(norms, np.ones(10))


def test_enet_projection_norm():
    random_state = check_random_state(0)
    norms = np.zeros(10)
    for i in range(10):
        a = random_state.randn(20000) * 10
        c = enet_projection(a, 1, 0.15)
        norms[i] = enet_norm(c, l1_ratio=0.15)
    print(norms)
    assert_array_almost_equal(norms, np.ones(10))


def test_enet_projection():
    c = np.empty((10, 100))
    b = np.empty((10, 100))
    random_state = check_random_state(0)
    for i in range(10):
        a = random_state.randn(100)
        b[i, :] = enet_projection_slow(a, radius=1, l1_ratio=0.1)
        c[i] = enet_projection(a, radius=1, l1_ratio=0.1)
    assert_array_almost_equal(c, b, 4)


def test_fast_enet_l2_ball():
    random_state = check_random_state(0)
    norms = np.zeros(10)
    for i in range(10):
        a = random_state.randn(100)
        c = np.zeros(100)
        c[:] = enet_projection(a, 1, 0.0)
        norms[i] = np.sqrt(np.sum(c ** 2))
    assert_array_almost_equal(norms, np.ones(10))


def test_fast_enet_l1_ball():
    random_state = check_random_state(0)
    norms = np.zeros(10)
    for i in range(10):
        a = random_state.randn(100)
        b = np.zeros(100)
        b[:] = enet_projection(a, 1, 1.0)
        norms[i] = np.sum(np.abs(b))
    assert_array_almost_equal(norms, np.ones(10))
