"""B11: the widths and curvatures follow their own principal axes along the
beam (Eduard's suggestion), rather than being sorted at each point"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scotty.beam_me_up import beam_me_up
from scotty.fun_general import find_beam_widths_and_curvs, follow_principal_axes
from scotty.init_bruv import get_parameters_for_Scotty


def eigh(matrices):
    return np.linalg.eigh(np.asarray(matrices))


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


# ------------------------------------------------------------ the matching


def test_crossing_eigenvalues_keep_their_axes():
    """diag(1 + t, 2 - t): the eigenvalues cross at t = 0.5. Sorted, each has
    a cusp there; followed, each stays on its axis and is a straight line"""
    t = np.linspace(0.0, 2.0, 201)
    matrices = np.zeros((len(t), 2, 2))
    matrices[:, 0, 0] = 1 + t
    matrices[:, 1, 1] = 2 - t
    eigvals, eigvecs, poor, _ = follow_principal_axes(*eigh(matrices))

    # Labelled where they differ most (t = 2: 3 against 0), smaller first
    assert_allclose(eigvals[:, 0], 2 - t, atol=1e-12)
    assert_allclose(eigvals[:, 1], 1 + t, atol=1e-12)
    assert_allclose(np.abs(eigvecs[:, 1, 0]), 1.0)  # along the second axis
    assert not poor.any()
    # The sorted ones turn a corner instead
    sorted_first = np.linalg.eigvalsh(matrices)[:, 0]
    assert np.abs(np.diff(sorted_first, 2)).max() > 1e-3
    assert np.abs(np.diff(eigvals[:, 0], 2)).max() < 1e-12


def test_avoided_crossing_follows_the_turning_axis():
    """[[t, e], [e, -t]]: the eigenvalues come within 2e of each other and
    turn away, while the axes turn by 90 degrees. Each eigenvalue follows its
    turning axis, which is the sorted order"""
    t = np.linspace(-1.0, 1.0, 401)
    e = 0.05
    matrices = np.array([[[x, e], [e, -x]] for x in t])
    eigvals, eigvecs, poor, _ = follow_principal_axes(*eigh(matrices))
    assert_allclose(eigvals, np.linalg.eigvalsh(matrices), atol=1e-12)
    assert not poor.any()
    # The axes turn continuously, with no change of sign
    assert np.all(np.sum(eigvecs[1:, :, 0] * eigvecs[:-1, :, 0], axis=1) > 0.9)


def test_rotating_ellipse_keeps_its_eigenvalues_and_turns_smoothly():
    angles = np.linspace(0.0, np.pi, 200)
    matrices = np.array(
        [rotation(a) @ np.diag([1.0, 2.0]) @ rotation(a).T for a in angles]
    )
    eigvals, eigvecs, poor, _ = follow_principal_axes(*eigh(matrices))
    assert_allclose(eigvals[:, 0], 1.0)
    assert_allclose(eigvals[:, 1], 2.0)
    first_axis = eigvecs[:, :, 0]
    assert np.all(np.sum(first_axis[1:] * first_axis[:-1], axis=1) > 0)
    # It ends turned by 180 degrees: the same line, followed round
    assert_allclose(first_axis[-1], -first_axis[0], atol=1e-12)
    assert not poor.any()


@pytest.mark.parametrize("coupling", [0.01, 0.03])
def test_coarsely_sampled_avoided_crossing_is_not_taken_for_a_crossing(coupling):
    """The referee's case (2026-11-16): with steps of 0.1 the axes turn by
    about 85 (coupling 0.01) or 73 degrees (0.03) over two steps, which
    matched by the axes alone looks like a crossing with a small turn. But
    the off-diagonal element in the previous axes stays about as large as the
    coupling, so it isn't one: the order is kept, as sorted, at every step"""
    for step in (0.1, 0.01, 0.0005):
        t = np.arange(-1.0, 1.0 + step / 2, step) + step / 3
        matrices = np.array([[[3 + x, coupling], [coupling, 3 - x]] for x in t])
        eigvals, _, _, _ = follow_principal_axes(*eigh(matrices))
        # Where the gap, relative to the sum of the eigenvalues, is over 1%
        decided = np.hypot(t, coupling) / 3 > 1e-2
        assert_allclose(
            eigvals[decided], np.linalg.eigvalsh(matrices)[decided], atol=1e-12
        )


def test_coarsely_sampled_crossing_is_still_a_crossing():
    """A true crossing doesn't turn the axes, however coarse the steps"""
    t = np.linspace(0.0, 2.0, 6)
    matrices = np.zeros((len(t), 2, 2))
    matrices[:, 0, 0] = 1 + t
    matrices[:, 1, 1] = 2 - t
    eigvals, _, poor, _ = follow_principal_axes(*eigh(matrices))
    assert_allclose(eigvals[:, 0], 2 - t, atol=1e-12)
    assert not poor.any()


def test_circular_stretch_is_carried_through():
    """Equal eigenvalues (a circular beam) for a while, where the axes are
    undefined, then splitting: no error, nothing flagged"""
    split = np.concatenate([np.zeros(50), np.linspace(0, 1, 50)])
    matrices = np.array([np.diag([1.0, 1.0 + s]) for s in split])
    eigvals, _, poor, _ = follow_principal_axes(*eigh(matrices))
    assert_allclose(eigvals[:, 1], 1.0 + split)
    assert not poor.any()


def test_noisy_nearly_circular_stretch_does_not_set_the_order():
    """The referee's case (2026-11-16): split, then nearly circular with noise
    (relative gap 2e-4, random axes) for 300 points, then split again along
    the same axes. The noise mustn't decide the order after the stretch"""
    split = np.concatenate(
        [np.linspace(0.5, 1e-3, 100), np.zeros(300), np.linspace(1e-3, 0.5, 100)]
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        matrices = []
        for s in split:
            matrix = np.diag([1.0, 1.0 + s])
            if s == 0:
                axis = rotation(rng.uniform(0, np.pi))[:, 0]
                matrix = matrix + 4e-4 * np.outer(axis, axis)
            matrices.append(matrix)
        matrices = np.array(matrices)
        eigvals, _, poor, _ = follow_principal_axes(*eigh(matrices))
        # Where the order is decided (gap over 1%), the same axes as before
        decided = split > 0.03
        assert_allclose(eigvals[decided], np.linalg.eigvalsh(matrices[decided]))
        assert not poor.any()


def test_circular_everywhere_is_left_alone():
    matrices = np.array([np.eye(2)] * 10)
    eigvals, _, poor, _ = follow_principal_axes(*eigh(matrices))
    assert_allclose(eigvals, 1.0)
    assert not poor.any()


def test_axes_turning_too_far_between_points_are_flagged():
    """44 degrees between points: more than 40, too far to follow reliably"""
    angles = np.radians(44.0) * np.arange(6)
    matrices = np.array(
        [rotation(a) @ np.diag([1.0, 2.0]) @ rotation(a).T for a in angles]
    )
    _, _, poor, _ = follow_principal_axes(*eigh(matrices))
    # All but the point where the labels are set (any one here, since the
    # eigenvalues are the same everywhere), which has nothing to match
    assert poor.sum() == len(angles) - 1


def test_single_matrix():
    """A single beam matrix gives four numbers, as before"""
    Psi_w = np.array([[1.0 + 2.0j, 0.0], [0.0, 3.0 + 4.0j]])
    K = np.array([1.0, 0.0, 0.0])
    curv1, curv2, width1, width2 = find_beam_widths_and_curvs(Psi_w, K, K)
    assert_allclose([curv1, curv2], [1.0, 3.0])
    assert_allclose([width1, width2], [1.0, np.sqrt(0.5)])


# ------------------------------------------------------------ in a run


def synthetic(tmp_path, **changes):
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.update(output_path=tmp_path, **changes)
    return kwargs


@pytest.fixture(scope="module")
def crossing_run(tmp_path_factory):
    """With no poloidal field and no toroidal launch angle, the beam has a
    plane of symmetry, Psi_xy vanishes, and the widths really cross"""
    return beam_me_up(**synthetic(tmp_path_factory.mktemp("crossing"), B_p_a=0.0))


def test_widths_cross_smoothly_in_a_run(crossing_run):
    beam = crossing_run["beam"].ds
    w1, w2 = beam["width_1"].values, beam["width_2"].values
    sign = np.sign(w1 - w2)
    assert np.count_nonzero(np.diff(sign[sign != 0])) >= 1  # they cross
    # Sorted, the larger width turns a corner at the crossing; followed, it
    # doesn't (Kai, 2026-11-16: 40 times smaller second difference)
    lam = np.linalg.eigvalsh(np.imag(beam["Psi_w"].values))
    sorted_larger = np.sqrt(2 / lam[:, 0])
    assert np.abs(np.diff(w1, 2)).max() < 0.2 * np.abs(np.diff(sorted_larger, 2)).max()


def test_axes_are_unit_transverse_and_continuous(crossing_run):
    beam = crossing_run["beam"].ds
    for quantity in ("width", "curvature"):
        g_hat = beam["g_hat_cartesian"].values
        first = beam[f"{quantity}_1_direction"].values
        second = beam[f"{quantity}_2_direction"].values
        # NaN only where the two are equal and the axes undefined
        defined = np.isfinite(first).all(axis=1)
        assert defined.mean() > 0.9
        first, second, g_hat = first[defined], second[defined], g_hat[defined]
        assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-12)
        assert_allclose(np.sum(first * second, axis=1), 0.0, atol=1e-12)
        assert_allclose(np.sum(first * g_hat, axis=1), 0.0, atol=1e-12)
        assert np.all(np.sum(first[1:] * first[:-1], axis=1) > 0)


def assert_sorted_where_different(first, second):
    """Where the two are nearly equal (a nearly circular beam), their order
    changes nothing visible: only where they differ by more than 1% (as
    `PRINCIPAL_AXES_GAP_MATTERS`, relative to their sizes, but not to less
    than 1% of the largest, as where curvatures pass through zero) is it
    checked"""
    first, second = first.values, second.values
    size = np.abs(first) + np.abs(second)
    size = np.maximum(size, 1e-2 * size.max())
    different = np.abs(second - first) > 1e-2 * size
    assert different.mean() > 0.5
    assert np.all(first[different] < second[different])


def logged(tmp_path):
    """`beam_me_up` clears the root logger's handlers, caplog's among them, so
    warnings are read from the run's log file"""
    (log_file,) = tmp_path.glob("scotty_log*.log")
    return log_file.read_text(encoding="utf-8")


def test_default_run_unchanged(tmp_path):
    """DBS_synthetic as given (Psi_xy is not zero): its widths and curvatures
    never cross, so the labels are the sorted ones, as before, with no
    warning, wherever the order matters"""
    dt = beam_me_up(**synthetic(tmp_path, file_log_level="WARNING"))
    assert "principal axes" not in logged(tmp_path)
    beam = dt["beam"].ds
    # Sorted (width_1 the larger) wherever they differ by more than 1%; near
    # the launch, where the beam is nearly circular, they follow the axes
    assert_sorted_where_different(beam["width_2"], beam["width_1"])
    # The curvatures, too, in the sorted order everywhere
    assert_sorted_where_different(beam["curvature_1"], beam["curvature_2"])


def test_coarse_run_keeps_sorted_order(tmp_path):
    """The referee's case (2026-11-16): launched numerically in vacuum, the
    curvatures come within about 4% of each other just past the cut-off, and
    their axes turn by tens of degrees between output points at the default
    len_tau. They keep the sorted order there. Whether a step turns by more
    than 40 degrees, and so is reported, depends on where the output points
    fall relative to the near-crossing (59 degrees on Windows; not reported on
    the Linux CI), so only the order is checked; the warning is checked in
    the next test, where the axes turn by 90 degrees within one step"""
    dt = beam_me_up(
        **synthetic(tmp_path, launch_flag="vacuum_numerical", Psi_BC_flag=None)
    )
    beam = dt["beam"].ds
    assert_sorted_where_different(beam["curvature_1"], beam["curvature_2"])


def test_weakly_broken_symmetry_is_an_avoided_crossing(tmp_path):
    """With a weak poloidal field (B_p_a = 0.01), the curvatures come within
    2% of each other just past the cut-off, and their axes turn by 90 degrees
    within one output step: an avoided crossing, not resolved at the default
    len_tau, which matched by the axes alone looks like a crossing with a 7
    degree turn. But in the previous axes the off-diagonal element is 2e-3 of
    the size, against 1e-8 to 1e-6 at a true crossing (B_p_a = 0). So the
    order is kept, as sorted, and the run says so (the referee's case,
    2026-11-16)"""
    dt = beam_me_up(**synthetic(tmp_path, B_p_a=0.01, file_log_level="WARNING"))
    assert "principal axes of the beam's curvatures" in logged(tmp_path)
    beam = dt["beam"].ds
    assert_sorted_where_different(beam["curvature_1"], beam["curvature_2"])
