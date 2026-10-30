"""Cylindrical -> Cartesian conversion of the Hessians of H (hamiltonian.convert_hessians).

Test Hamiltonian, after Aurora's check (students/aurora/projects/004-scotty4-analysis/
scripts/check_convert_hessians.py): H(x, K) = K.K - n(R, Z) + c (b(R, Z).K)^2 with a
sheared, position-dependent b, so that it is axisymmetric (dH/dzeta = 0, as Scotty
assumes) but every block of the Hessian is non-trivial. The exact Cartesian Hessians
are central differences of H in (X, Y, Z, K_X, K_Y, K_Z); the cylindrical inputs are
central differences of H(x(q), K(q, p)) in the canonical (R, zeta, Z, K_R, K_zeta, K_Z),
with K_zeta the dimensionless mode number, in the layout Scotty uses.

At fixed Cartesian K, K_R and K_zeta depend on position, so the spatial and mixed
blocks pick up terms in dH/dK_R, dH/dK_zeta and the mixed cylindrical derivatives.
Leaving those out gave O(1) errors (4.0 and 7.7 relative) in the two blocks.
"""

from scotty.hamiltonian import convert_hessians

import numpy as np
import pytest
from numpy.testing import assert_allclose

C_ANISO = 0.3


def H_cart(x, K):
    R, Z = np.hypot(x[0], x[1]), x[2]
    zeta = np.arctan2(x[1], x[0])
    n = 1.0 + 0.4 * (R - 1.5) ** 2 + 0.2 * Z**2 + 0.1 * R * Z
    BR, BT, BZ = 0.2 * Z, 1.0 / R, 0.3 * (R - 1.5)
    b = np.array([BR * np.cos(zeta) - BT * np.sin(zeta), BR * np.sin(zeta) + BT * np.cos(zeta), BZ])
    b = b / np.linalg.norm(b)
    return K @ K - n + C_ANISO * (b @ K) ** 2


def to_cart(q, p):
    R, zeta, Z = q
    K_R, K_zeta, K_Z = p
    x = np.array([R * np.cos(zeta), R * np.sin(zeta), Z])
    K = np.array(
        [
            K_R * np.cos(zeta) - K_zeta / R * np.sin(zeta),
            K_R * np.sin(zeta) + K_zeta / R * np.cos(zeta),
            K_Z,
        ]
    )
    return x, K


def gradient_and_hessian(f, v, h=1e-4):
    n = len(v)
    g = np.empty(n)
    Hm = np.empty((n, n))
    for i in range(n):
        ei = np.eye(n)[i] * h
        g[i] = (f(v + ei) - f(v - ei)) / (2 * h)
        for j in range(n):
            ej = np.eye(n)[j] * h
            Hm[i, j] = (f(v + ei + ej) - f(v + ei - ej) - f(v - ei + ej) + f(v - ei - ej)) / (4 * h * h)
    return g, Hm


def random_points(n, seed=3):
    rng = np.random.default_rng(seed)
    q = np.column_stack([rng.uniform(1.2, 2.0, n), rng.uniform(-np.pi, np.pi, n), rng.uniform(-0.4, 0.4, n)])
    p = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n) * q[:, 0], rng.uniform(-1, 1, n)])
    return q, p


def cylindrical_inputs_and_exact(q, p):
    """Scotty-layout cylindrical derivatives at each point, and the exact Cartesian Hessians"""
    N = len(q)
    g = np.empty((N, 6))
    H_cyl = np.empty((N, 6, 6))
    H_exact = np.empty((N, 6, 6))
    K_cart = np.empty((N, 3))
    for i in range(N):
        g[i], H_cyl[i] = gradient_and_hessian(lambda v: H_cart(*to_cart(v[:3], v[3:])), np.concatenate([q[i], p[i]]))
        x, K = to_cart(q[i], p[i])
        K_cart[i] = K
        _, H_exact[i] = gradient_and_hessian(lambda w: H_cart(w[:3], w[3:]), np.concatenate([x, K]))
    dH = dict(dH_dR=g[:, 0], dH_dzeta=g[:, 1], dH_dZ=g[:, 2], dH_dKR=g[:, 3], dH_dKzeta=g[:, 4], dH_dKZ=g[:, 5])
    return dH, H_cyl, H_exact, K_cart


def test_test_hamiltonian_is_axisymmetric():
    q, p = random_points(5)
    dH, *_ = cylindrical_inputs_and_exact(q, p)
    assert np.max(np.abs(dH["dH_dzeta"])) < 1e-6


@pytest.mark.parametrize("n_points", [10, 1])
def test_cylindrical_to_cartesian_hessians(n_points):
    """All three converted blocks match the exact Cartesian Hessians, for a batch
    of points and for a single point (the N = 1 path squeezes the batch axis)"""
    q, p = random_points(n_points)
    dH, H_cyl, H_exact, K_cart = cylindrical_inputs_and_exact(q, p)
    if n_points == 1:
        args = (q[0], K_cart[0], {k: v[0] for k, v in dH.items()}, H_cyl[0, :3, :3], H_cyl[0, 3:, :3], H_cyl[0, 3:, 3:])
        expected = (H_exact[0, :3, :3], H_exact[0, 3:, :3], H_exact[0, 3:, 3:])
    else:
        args = (q.T, K_cart.T, dH, H_cyl[:, :3, :3], H_cyl[:, 3:, :3], H_cyl[:, 3:, 3:])
        expected = (H_exact[:, :3, :3], H_exact[:, 3:, :3], H_exact[:, 3:, 3:])

    got = convert_hessians(*args, start="cylindrical", end="cartesian")

    for name, g_, e_ in zip(("grad_grad", "gradK_grad", "gradK_gradK"), got, expected):
        scale = np.max(np.abs(e_))
        assert_allclose(g_, e_, rtol=0, atol=1e-5 * scale, err_msg=name)


def test_cartesian_inputs_pass_through():
    """With start == end == cartesian the Hessians are returned unchanged"""
    q, p = random_points(3)
    _, H_cyl, _, K_cart = cylindrical_inputs_and_exact(q, p)
    blocks = (H_cyl[:, :3, :3], H_cyl[:, 3:, :3], H_cyl[:, 3:, 3:])
    got = convert_hessians(q.T, K_cart.T, {}, *blocks, start="cartesian", end="cartesian")
    for g_, b_ in zip(got, blocks):
        assert_allclose(g_, b_)
