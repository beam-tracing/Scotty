"""Structure and consistency of the analysis outputs.

Regression tests for bugs where an output was silently lost or had the wrong
type: a dict key written twice (the second value replacing the first), a
variable defined only in the beam-tracing branch, complex-typed filter functions,
and the polarisation vector taken from the wrong eigenvector.
"""

from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

import numpy as np
import pytest
from scipy.integrate import trapezoid


def _run(tmp_path_factory, **overrides):
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["output_path"] = tmp_path_factory.mktemp("analysis_outputs")
    kwargs_dict.update(overrides)
    return beam_me_up(**kwargs_dict)


@pytest.fixture(scope="module", params=[1, -1], ids=["O", "X"])
def beam_run(request, tmp_path_factory):
    return _run(tmp_path_factory, mode_flag=request.param)


@pytest.fixture(scope="module")
def ray_run(tmp_path_factory):
    return _run(tmp_path_factory, ray_tracing_flag=True)


def test_ray_tracing_only_analysis_completes(ray_run):
    """delta_ne_sq needs the launch wavenumber, which used to be set only for beam
    tracing, so a ray-tracing-only run failed in dbs_analysis"""
    dbs = ray_run["diagnostics/dbs"]
    for power in ("-10_3", "-13_3"):
        assert np.all(np.isfinite(dbs[f"delta_ne_sq_{power}"].values))


def test_filterfun_all_is_an_array_with_its_own_integral(beam_run):
    """filterfun_all is the product of the four pieces of the filter function,
    without the turbulence spectrum delta_ne_sq. The array and its path integral
    used to share a key, and the 0-d integral replaced the array"""
    dbs = beam_run["diagnostics/dbs"]
    length = dbs["beam_path_length"].values
    filterfun_all = dbs["filterfun_all"].values
    assert filterfun_all.shape == length.shape
    product = np.prod(
        [
            dbs[k].values
            for k in ("filterfun_p", "filterfun_r", "filterfun_b", "filterfun_m")
        ],
        axis=0,
    )
    np.testing.assert_allclose(filterfun_all, product, rtol=1e-12)
    np.testing.assert_allclose(
        dbs["int_filterfun_all_dl"].values,
        trapezoid(filterfun_all, length),
        rtol=1e-12,
    )


def test_d_xhat_d_tau_keeps_vector_and_scalar(beam_run):
    """The scalar d_xhat_d_tau . y_hat used to overwrite the vector"""
    analysis = beam_run["beam"]
    vector = analysis["d_xhat_d_tau"].values
    assert vector.shape == analysis["x_hat_cartesian"].values.shape
    np.testing.assert_allclose(
        analysis["d_xhat_d_tau_dot_yhat"].values,
        np.einsum("ti,ti->t", vector, analysis["y_hat_cartesian"].values),
        rtol=0,
        atol=1e-12 * np.max(np.abs(vector)),
    )


def test_filter_functions_are_real(beam_run):
    """Rounding in the complex Cardano formula used to make filterfun_r, filterfun_all and
    their integrals complex-typed"""
    analysis, dbs = beam_run["beam"], beam_run["diagnostics/dbs"]
    assert not np.iscomplexobj(beam_run["beam"]["g_magnitude_Cardano"].values)
    for k in (
        "filterfun_r",
        "filterfun_all",
        "int_filterfun_r_dl",
        "int_filterfun_all_dl",
    ):
        assert not np.iscomplexobj(dbs[k].values), k


def test_e_hat_belongs_to_the_traced_mode(beam_run):
    """e_hat is the eigenvector whose eigenvalue is the Cardano root the beam
    follows (eigh sorts eigenvalues, so the solver's Cardano index cannot be
    used to pick it)"""
    analysis = beam_run["beam"]
    np.testing.assert_allclose(
        beam_run["beam"]["H_eigval_solution"].values,
        beam_run["beam"]["H_Cardano_solution"].values,
        rtol=0,
        atol=1e-10,
    )


def test_k_perp1_bs_splits_into_normal_and_binormal(beam_run):
    """n_hat = grad psi / |grad psi| is perpendicular to b_hat (B lies in the
    flux surface), and so is k_perp1_hat, so {n_hat, b_hat x n_hat} spans the
    plane of k_perp1_hat: the two components add in quadrature to k_perp1_bs.
    At the cut-off K is tangent to the flux surface, so the backscattered
    wavevector is almost purely binormal (the usual DBS picture)"""
    analysis, dbs = beam_run["beam"], beam_run["diagnostics/dbs"]
    b_hat = analysis["b_hat_cartesian"].values
    normal_hat = analysis["normal_hat"].values
    np.testing.assert_allclose(
        np.einsum("ti,ti->t", normal_hat, b_hat), 0, rtol=0, atol=1e-6
    )
    k_total = dbs["k_perp1_bs"].values
    k_normal = dbs["k_perp1_bs_normal"].values
    k_binormal = dbs["k_perp1_bs_binormal"].values
    np.testing.assert_allclose(
        np.hypot(k_normal, k_binormal), np.abs(k_total), rtol=1e-6
    )
    cutoff = int(analysis["cutoff_index"])
    assert abs(k_normal[cutoff]) < 0.05 * abs(k_total[cutoff])


def test_k_perp2_bs(beam_run):
    """k_perp2_bs is along y_hat, which is perpendicular to b_hat, so its normal
    and binormal components add in quadrature to it too. It is the k_perp2 at
    which the k_perp2 integrand peaks, 2 K theta_m Im(M_xy^-1) / Im(M_yy^-1), so
    it is zero at zero mismatch. On this unsteered case the mismatch is small, and
    at the cut-off k_perp2_bs is well inside the k_perp2 resolution. (Away from
    the cut-off the resolution changes quickly in X mode, so the ratio is only
    checked there.)"""
    dbs = beam_run["diagnostics/dbs"]
    cutoff = int(beam_run["beam"]["cutoff_index"])
    k_total = dbs["k_perp2_bs"].values
    np.testing.assert_allclose(
        np.hypot(dbs["k_perp2_bs_normal"].values, dbs["k_perp2_bs_binormal"].values),
        np.abs(k_total),
        rtol=1e-6,
    )
    assert abs(k_total[cutoff]) < 0.1 * dbs["delta_k_perp2"].values[cutoff]
