r"""The root of the Booker quadratic in H (fun_general.find_Booker_root).

H = N^2 + (beta - mode_flag sqrt(disc)) / (2 alpha) is 0/0 for one root at
alpha = epsilon_perp = 0, the X-mode's upper hybrid layer. For K perpendicular
to B, that is where Omega_pe^2 / Omega^2 = 1 - (Omega_ce / Omega)^2. The O-mode
is regular there (N^2 = 1 - Omega_pe^2 / Omega^2), so its H must stay zero
through the layer, and the X-mode must keep its resonance.

`find_Booker_root` is the one from beam-tracing/Scotty#181. These tests came with
#173, whose own version used the rationalised form where it is the smaller in
magnitude; for a positive discriminant that is the same condition as
mode_flag * beta > 0, and the two gave identical values on every input tried.
"""

import importlib
import time
from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import constants

from scotty.boundary_conditions import find_H_bar
from scotty.fun_general import (
    find_Booker_discriminant,
    find_Booker_root,
    find_Booker_terms,
)
from scotty.geometry import MagneticField_Cartesian
from scotty.hamiltonian import Hamiltonian
from scotty.launch import _mode_polarisations
from scotty.logger import config_logger

OMEGA = 2 * np.pi * 105e9
K0 = OMEGA / constants.c
GYRO_RATIO = 0.3  # Omega_ce / Omega
B0 = GYRO_RATIO * OMEGA * constants.m_e / constants.e
# The density at which Omega_pe^2 / Omega^2 = 1, in 10^19 m^-3
N_CUTOFF = constants.epsilon_0 * constants.m_e * OMEGA**2 / constants.e**2 / 1e19
LAYER = 1 - GYRO_RATIO**2  # Omega_pe^2 / Omega^2 at alpha = 0, for K perp B
# Relative distances from the layer, down to rounding
OFFSETS = np.array(
    [0.0, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
)


@pytest.fixture(scope="module", autouse=True)
def trace_level(tmp_path_factory):
    """`Hamiltonian.__call__` uses `log.trace`, a level `config_logger` registers"""
    config_logger("warning", None, tmp_path_factory.mktemp("log"), "")


def old_root(alpha, beta, gamma, discriminant, mode_flag):
    """The form before `find_Booker_root`"""
    return (beta - mode_flag * np.sqrt(discriminant)) / (2 * alpha)


def uniform_field():
    """Uniform B = -B0 Z_hat, with poloidal flux = X. The density below makes X
    the ratio Omega_pe^2 / Omega^2"""
    field = MagicMock(spec=MagneticField_Cartesian)
    field.polflux = lambda X, Y, Z: X
    field.magnitude = lambda X, Y, Z: B0 * np.ones_like(X)
    field.unitvector = lambda X, Y, Z: np.array(
        [0 * np.ones_like(X), 0 * np.ones_like(X), -np.ones_like(X)]
    ).T
    field.all = lambda X, Y, Z: (
        B0 * field.unitvector(X, Y, Z),
        field.unitvector(X, Y, Z),
        field.magnitude(X, Y, Z),
    )
    return field


def hamiltonian(mode_flag):
    return Hamiltonian(
        launch_angular_freq=OMEGA,
        mode_flag=mode_flag,
        deltas=np.array([1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1]),
        field=uniform_field(),
        density_fit=lambda polflux: polflux * N_CUTOFF,
    )


def booker(density_ratio, sin_theta_m_sq=0.0, B=B0):
    args = (OMEGA, B, sin_theta_m_sq, density_ratio * N_CUTOFF)
    return *find_Booker_terms(*args), find_Booker_discriminant(*args)


@pytest.mark.parametrize("side", (-1, 1))
def test_O_mode_H_through_the_layer(side):
    """For K perp B on the O-mode's dispersion relation, H is zero to rounding at
    and near alpha = 0, on both sides of it. This is 1 - Omega_pe^2 / Omega^2"""
    density_ratio = LAYER * (1 + side * OFFSETS)
    K = K0 * np.sqrt(1 - density_ratio)
    zero = np.zeros_like(K)
    H = hamiltonian(1)(density_ratio, zero, zero, K, zero, zero)
    assert np.max(np.abs(H)) < 2e-15


def test_O_mode_root_is_one_minus_density_ratio():
    """The root itself, over the whole range below the cut-off, through the layer"""
    density_ratio = np.concatenate(
        [np.linspace(0.0, 0.99, 100), LAYER * (1 + OFFSETS), LAYER * (1 - OFFSETS)]
    )
    root = find_Booker_root(*booker(density_ratio), 1)
    assert_allclose(root, density_ratio - 1, rtol=0, atol=1e-15)


def test_X_mode_keeps_its_resonance():
    """For K perp B, N^2 = ((1 - X)^2 - Y^2) / (1 - X - Y^2) with
    X = Omega_pe^2 / Omega^2 and Y = Omega_ce / Omega, which goes through the
    upper hybrid resonance with a change of sign. That formula's own denominator
    loses digits close to the layer, so it is only compared up to 1e-4 from it"""
    density_ratio = np.concatenate(
        [
            np.linspace(0.01, 0.9, 50),
            np.linspace(0.92, 1.5, 50),
            LAYER * (1 + OFFSETS[-2:]),
            LAYER * (1 - OFFSETS[-2:]),
        ]
    )
    root = find_Booker_root(*booker(density_ratio), -1)
    X, Y = density_ratio, GYRO_RATIO
    assert_allclose(-root, ((1 - X) ** 2 - Y**2) / (1 - X - Y**2), rtol=1e-11)
    just_below, just_above = -find_Booker_root(
        *booker(LAYER * (1 + np.array([-1e-8, 1e-8]))), -1
    )
    assert just_below < -1e6 and just_above > 1e6


def test_exactly_zero_alpha():
    """alpha = 0, set by hand: the regular root is gamma / beta with no warning,
    and the other root is the X-mode's infinity"""
    alpha, beta, gamma = 0.0, 0.09, -0.0081
    discriminant = beta**2
    with np.errstate(all="raise"):
        assert_allclose(
            find_Booker_root(alpha, beta, gamma, discriminant, 1), gamma / beta
        )
    with np.errstate(divide="ignore"):
        assert np.isinf(find_Booker_root(alpha, beta, gamma, discriminant, -1))


@pytest.mark.parametrize("mode_flag", (1, -1))
def test_old_and_new_forms_agree_away_from_alpha_zero(mode_flag):
    """Random physical points, as in test_booker_discriminant, away from alpha = 0.
    Where the numerator is the larger of the two the form is untouched, to the
    last bit. Elsewhere the two differ only by the rounding of the old form's
    cancellation"""
    rng = np.random.default_rng(1)
    n = 20000
    gyro_ratio = rng.uniform(0.01, 2.5, n)
    density_ratio = 10 ** rng.uniform(-7, 1.5, n)
    sin_sq = rng.uniform(0.0, 1.0, n)
    terms = booker(
        density_ratio, sin_sq, gyro_ratio * OMEGA * constants.m_e / constants.e
    )
    alpha, beta, gamma, discriminant = terms
    far = np.abs(alpha) > 1e-3
    assert np.count_nonzero(far) > 0.9 * n

    old = old_root(*terms, mode_flag)
    new = find_Booker_root(*terms, mode_flag)
    root_disc = np.sqrt(discriminant)
    large = np.abs(beta - mode_flag * root_disc) >= np.abs(beta + mode_flag * root_disc)
    assert 0 < np.count_nonzero(large) < n
    assert np.array_equal(old[large], new[large])

    rounding = 1e-15 * (np.abs(beta) + root_disc) / (2 * np.abs(alpha))
    difference = np.abs(old - new)
    assert np.all(difference[far] <= 4 * rounding[far] + 1e-15 * np.abs(new[far]))


def test_shapes_and_elements_choose_independently():
    """A scalar gives a scalar, and in an array each element picks its own form:
    here the first is on the rationalised side, and the second on the other"""
    assert np.ndim(find_Booker_root(1.0, -2.0, 1.0, 0.0, 1)) == 0
    assert_allclose(find_Booker_root(1.0, -2.0, 1.0, 0.0, 1), -1.0)

    alpha = np.array([0.5, 0.5])
    beta = np.array([1.0, -1.0])
    gamma = np.array([0.1, 0.1])
    discriminant = beta**2 - 4 * alpha * gamma
    new = find_Booker_root(alpha, beta, gamma, discriminant, 1)
    assert new.shape == (2,)
    assert_allclose(new, old_root(alpha, beta, gamma, discriminant, 1), rtol=1e-14)
    assert new[1] == old_root(alpha, beta, gamma, discriminant, 1)[1]


def test_nan_stays_nan():
    """A NaN density makes all the Booker terms NaN, and the root stays NaN in that
    element only. `find_mode_index_Cardano` (analysis.py) and `check_H_output` rely
    on seeing them"""
    terms = booker(np.array([np.nan, 0.5, 0.5]))
    for mode_flag in (1, -1):
        root = find_Booker_root(*terms, mode_flag)
        assert np.isnan(root[0])
        assert np.isfinite(root[1:]).all()

    zero = np.zeros(1)
    H = hamiltonian(1)(np.array([np.nan]), zero, zero, np.array([K0]), zero, zero)
    assert np.isnan(H).all()


def test_find_H_bar_through_the_layer():
    """The same root in the boundary conditions' H_bar = K^2 + K0^2 (root)"""
    x_hat, y_hat, z_hat = np.eye(3)
    for offset in OFFSETS:
        density_ratio = LAYER * (1 + offset)
        K_normal = K0 * np.sqrt(1 - density_ratio)
        H_bar = find_H_bar(
            mode_flag=1,
            K_parallel=0.0,
            K_binormal=0.0,
            K_normal=K_normal,
            parallel_unitvector=z_hat,
            binormal_unitvector=y_hat,
            normal_unitvector=x_hat,
            B_X=0.0,
            B_Y=0.0,
            B_Z=-B0,
            launch_angular_frequency=OMEGA,
            electron_density=density_ratio * N_CUTOFF,
        )
        assert abs(H_bar) < 2e-15 * K0**2


def test_polarisations_at_the_layer():
    """The same root in `launch._mode_polarisations`: at the layer, K perp B, the
    O-mode is still found polarised along b"""
    for offset in (-1e-2, -1e-10, 1e-10, 1e-2):
        polarisations = _mode_polarisations(
            OMEGA, B0, 0.0, LAYER * (1 + offset) * N_CUTOFF
        )
        assert abs(polarisations[1][-1]) > 0.999


def test_benchmark_Hamiltonian_call(monkeypatch, record_property):
    """Times a Hamiltonian call with the old form of the root against the new, in
    turn, as this machine's speed drifts over seconds. Only reports the timings
    (shown with ``pytest -s``, and in the JUnit XML), as timings depend on the
    machine"""
    module = importlib.import_module("scotty.hamiltonian")
    roots = {"old": old_root, "new": module.find_Booker_root}
    n_points, repeats = 10_000, 15
    density_ratio = np.linspace(0.01, 0.8, n_points)
    zero = np.zeros(n_points)
    K_X = K0 * np.sqrt(1 - density_ratio) * 0.9
    K_Y = K0 * np.sqrt(1 - density_ratio) * np.sqrt(1 - 0.81)
    points = (density_ratio, zero, zero, K_X, K_Y, zero)
    H = hamiltonian(1)
    terms = booker(density_ratio)
    # (what is timed, the arguments, calls per timing)
    jobs = {
        "array": (H, points, 20),
        "single": (H, tuple(p[:1] * 1.0 for p in points), 200),
    }

    timings = {}
    for _ in range(repeats):
        for label, root in roots.items():
            monkeypatch.setattr(module, "find_Booker_root", root)
            for job, (function, arguments, calls) in jobs.items():
                start = time.perf_counter()
                for _ in range(calls):
                    function(*arguments)
                duration = (time.perf_counter() - start) / calls
                key = (label, job)
                timings[key] = min(timings.get(key, np.inf), duration)
            start = time.perf_counter()
            for _ in range(200):
                root(*terms, 1)
            duration = (time.perf_counter() - start) / 200
            timings[label, "root"] = min(timings.get((label, "root"), np.inf), duration)

    for key, value in timings.items():
        record_property("_".join(key) + "_seconds", value)
    print(
        f"\nHamiltonian call, {n_points} points: old "
        f"{timings['old', 'array'] * 1e3:.3f} ms, new "
        f"{timings['new', 'array'] * 1e3:.3f} ms"
        f"\nHamiltonian call, 1 point: old {timings['old', 'single'] * 1e6:.1f} us, "
        f"new {timings['new', 'single'] * 1e6:.1f} us"
        f"\nRoot only, {n_points} points: old {timings['old', 'root'] * 1e6:.1f} us, "
        f"new {timings['new', 'root'] * 1e6:.1f} us"
    )
    assert np.isfinite(timings["new", "array"])
