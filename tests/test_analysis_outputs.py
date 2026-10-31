"""Structure and consistency of the analysis outputs.

Regression tests for bugs where an output was silently lost or had the wrong
type: a dict key written twice (the second value replacing the first), a
variable defined only in the beam-tracing branch, complex-typed localisation,
and the polarisation vector taken from the wrong eigenvector.
"""

from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

import numpy as np
import pytest
from scipy.integrate import trapezoid


def _run(tmp_path_factory, **overrides):
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["figure_flag"] = False
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
    """loc_s needs the launch wavenumber, which used to be set only for beam
    tracing, so a ray-tracing-only run failed in dbs_analysis"""
    dbs = ray_run["dbs_analysis"]
    for power in ("-10_3", "-13_3"):
        assert np.all(np.isfinite(dbs[f"loc_s_{power}"].values))


def test_loc_all_is_an_array_with_its_own_integral(beam_run):
    """The array and its path integral used to share the key loc_all_<power>,
    and the 0-d integral replaced the array"""
    dbs = beam_run["dbs_analysis"]
    length = dbs["beam_path_length"].values
    for power in ("-10_3", "-13_3"):
        loc_all = dbs[f"loc_all_{power}"].values
        assert loc_all.shape == length.shape
        product = np.prod(
            [
                dbs[k].values
                for k in ("loc_p", "loc_r", "loc_b", "loc_m", f"loc_s_{power}")
            ],
            axis=0,
        )
        np.testing.assert_allclose(loc_all, product, rtol=1e-12)
        np.testing.assert_allclose(
            dbs[f"int_loc_all_{power}_dL"].values,
            trapezoid(loc_all, length),
            rtol=1e-12,
        )


def test_d_xhat_d_tau_keeps_vector_and_scalar(beam_run):
    """The scalar d_xhat_d_tau . y_hat used to overwrite the vector"""
    analysis = beam_run["analysis"]
    vector = analysis["d_xhat_d_tau"].values
    assert vector.shape == analysis["x_hat_cartesian"].values.shape
    np.testing.assert_allclose(
        analysis["d_xhat_d_tau_dot_yhat"].values,
        np.einsum("ti,ti->t", vector, analysis["y_hat_cartesian"].values),
        rtol=0,
        atol=1e-12 * np.max(np.abs(vector)),
    )


def test_localisation_is_real(beam_run):
    """Rounding in the complex Cardano formula used to make loc_r, loc_all and
    their integrals complex-typed"""
    analysis, dbs = beam_run["analysis"], beam_run["dbs_analysis"]
    assert not np.iscomplexobj(analysis["g_magnitude_Cardano"].values)
    for k in ("loc_r", "loc_all_-10_3", "int_loc_r_dL", "int_loc_all_-10_3_dL"):
        assert not np.iscomplexobj(dbs[k].values), k


def test_e_hat_belongs_to_the_traced_mode(beam_run):
    """e_hat is the eigenvector whose eigenvalue is the Cardano root the beam
    follows (eigh sorts eigenvalues, so the solver's Cardano index cannot be
    used to pick it)"""
    analysis = beam_run["analysis"]
    np.testing.assert_allclose(
        analysis["H_eigval_solution"].values,
        analysis["H_Cardano_solution"].values,
        rtol=0,
        atol=1e-10,
    )


def test_k_perp1_bs_splits_into_normal_and_binormal(beam_run):
    """n_hat = grad psi / |grad psi| is perpendicular to b_hat (B lies in the
    flux surface), and so is k_perp1_hat, so {n_hat, b_hat x n_hat} spans the
    plane of k_perp1_hat: the two components add in quadrature to k_perp1_bs.
    At the cut-off K is tangent to the flux surface, so the backscattered
    wavevector is almost purely binormal (the usual DBS picture)"""
    analysis, dbs = beam_run["analysis"], beam_run["dbs_analysis"]
    b_hat = analysis["b_hat_cartesian"].values
    normal_hat = dbs["normal_hat"].values
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
