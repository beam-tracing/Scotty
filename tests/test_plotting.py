"""The matplotlib plots of scotty.plotting, on a small run's output, in both
geometries. The plotted central ray must be the beam path."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import same_color
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.plotting import (
    beam_path_cartesian,
    beam_path_cylindrical,
    plot_bounding_box,
    plot_dispersion_relation,
    plot_flux_surfaces,
    plot_poloidal_beam_path,
    plot_toroidal_beam_path,
)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    parameters = get_parameters_for_Scotty("DBS_synthetic")
    parameters.update(len_tau=22, console_log_level="WARNING")
    return beam_me_up(
        **parameters,
        output_path=tmp_path_factory.mktemp("plotting"),
        return_dt_field=True,
    )


def as_cartesian(dt: xr.DataTree) -> xr.DataTree:
    """The same output as a ``geometry="cartesian"`` run would store it: the
    solver output is stored in both coordinate systems either way, but the
    inputs' grid coordinates and the geometry differ"""
    inputs = dt["inputs"].to_dataset().drop_vars(["R", "Z", "row", "col"])
    R = dt["inputs"]["R"].values
    inputs = inputs.assign(geometry="cartesian").assign_coords(
        X=np.linspace(-R.max(), R.max(), 11),
        Y=np.linspace(-R.max(), R.max(), 11),
        Z=dt["inputs"]["Z"].values,
    )
    tree = {group: dt[group].to_dataset() for group in dt.children}
    tree["inputs"] = inputs
    return xr.DataTree.from_dict(tree)


def line(ax, label):
    return next(l for l in ax.get_lines() if l.get_label() == label)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("geometry", ["cylindrical", "cartesian"])
def test_beam_path_in_both_geometries(run, geometry):
    dt, _ = run
    if geometry == "cartesian":
        dt = as_cartesian(dt)
    so = dt["solver"]
    R, zeta, Z = beam_path_cylindrical(dt)
    assert_allclose(R, so["q_R_output"].values, rtol=1e-12)
    assert_allclose(zeta, so["q_zeta_output"].values, atol=1e-12)
    assert_allclose(Z, so["q_Z_output"].values, rtol=1e-12)
    X, Y, Z = beam_path_cartesian(dt)
    assert_allclose(X, so["q_output_cartesian"].values[:, 0], rtol=1e-12)
    assert_allclose(Y, so["q_output_cartesian"].values[:, 1], atol=1e-12)


@pytest.mark.parametrize("geometry", ["cylindrical", "cartesian"])
@pytest.mark.parametrize("with_field", [False, True])
def test_poloidal_beam_path(run, geometry, with_field):
    dt, field = run
    if geometry == "cartesian":
        dt = as_cartesian(dt)
    so = dt["solver"]
    ax = plot_poloidal_beam_path(dt, zoom=True, field=field if with_field else None)

    ray = line(ax, "Central (reference) ray")
    assert_allclose(ray.get_xdata(), so["q_R_output"].values, rtol=1e-12)
    assert_allclose(ray.get_ydata(), so["q_Z_output"].values, rtol=1e-12)
    vacuum = line(ax, "Central (reference) ray in vacuum")
    assert_allclose(vacuum.get_xdata()[0], so["q_launch_cylindrical"].values[0])
    # The beam's two edges, either side of the ray
    edges = [l for l in ax.get_lines() if l.get_linestyle() == "--"]
    assert len(edges) == 2
    R_plus, R_minus = (np.asarray(e.get_xdata()) for e in edges)
    assert np.all(np.isfinite(R_plus)) and np.all(np.isfinite(R_minus))
    assert ax.get_xlabel() == "R [m]"
    if with_field:
        assert ax.collections  # flux-surface contours


@pytest.mark.parametrize("geometry", ["cylindrical", "cartesian"])
def test_toroidal_beam_path(run, geometry):
    dt, field = run
    if geometry == "cartesian":
        dt = as_cartesian(dt)
    so = dt["solver"]
    ax = plot_toroidal_beam_path(dt, field=field)
    ray = line(ax, "Central (reference) ray")
    # From the launch position, then the beam path
    assert_allclose(ray.get_xdata()[0], so["q_launch_cartesian"].values[0])
    assert_allclose(
        ray.get_xdata()[1:], so["q_output_cartesian"].values[:, 0], rtol=1e-12
    )
    assert_allclose(
        ray.get_ydata()[1:], so["q_output_cartesian"].values[:, 1], atol=1e-12
    )
    # The last closed flux surface on the midplane: R = R_axis +- a
    circles = [l for l in ax.get_lines() if same_color(l.get_color(), "orange")]
    radii = sorted(np.hypot(c.get_xdata(), c.get_ydata()).mean() for c in circles)
    assert_allclose(radii, [1.0, 2.0], rtol=1e-3)


def test_dispersion_relation(run):
    dt, _ = run
    ax = plot_dispersion_relation(dt)
    lines = ax.get_lines()
    assert len(lines) == 6
    for l in lines:
        assert_allclose(l.get_xdata(), dt["beam"]["beam_path_length_wrt_cutoff"].values)
    assert_allclose(
        lines[3].get_ydata(), np.abs(dt["beam"]["H_Cardano"].values[:, 0]), rtol=1e-12
    )


def test_helpers_and_saving(run, tmp_path):
    dt, field = run
    ax = plot_bounding_box(dt)
    assert len(ax.collections) == 2  # hlines, vlines
    plot_flux_surfaces(field, ax=ax)
    filename = tmp_path / "poloidal.png"
    plot_poloidal_beam_path(dt, filename=filename)
    assert filename.is_file()


def test_ray_tracing_only_has_no_width(tmp_path):
    parameters = get_parameters_for_Scotty("DBS_synthetic")
    parameters.update(len_tau=12, ray_tracing_flag=True, console_log_level="WARNING")
    dt = beam_me_up(**parameters, output_path=tmp_path)
    ax = plot_poloidal_beam_path(dt)
    assert not [l for l in ax.get_lines() if l.get_linestyle() == "--"]
    assert_allclose(
        line(ax, "Central (reference) ray").get_xdata(),
        dt["solver"]["q_R_output"].values,
    )


def test_3d_plots_need_plotly_only_when_called():
    """plotly is optional: importing scotty.plotting works without it"""
    import importlib.util

    import scotty.plotting

    if importlib.util.find_spec("plotly") is None:
        with pytest.raises(ImportError, match="plotly"):
            scotty.plotting.make_fig_if_None()
