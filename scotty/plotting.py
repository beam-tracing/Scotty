# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Plotting.

- matplotlib, for an output `xarray.DataTree` in either geometry, as v3's
  plots: `plot_dispersion_relation`, `plot_poloidal_beam_path` and
  `plot_toroidal_beam_path`, with the helpers `plot_bounding_box` and
  `plot_flux_surfaces`. v4's output does not store the poloidal flux on a
  grid, so the flux surfaces are drawn only if the field is passed (e.g. from
  ``beam_me_up(..., return_dt_field=True)``).
- plotly, in 3D: `plot_3d_flux_surfaces`, `plot_3d_beam` and
  `plot_3d_beam_in_plasma`.

v3's other plots (beam widths, filter functions, 3D beam profile, Psi) are
not ported yet.

plotly is not one of Scotty's dependencies: install it for the 3D plots.
"""

# Postponed annotations, so that the plotly type hints need no plotly
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    import plotly.graph_objects as pgo
    import plotly.io as pio
except ImportError:  # only the 3D plots need plotly

    class _MissingPlotly:
        def __getattr__(self, name):
            raise ImportError(
                "The 3D plots in `scotty.plotting` need plotly (pip install plotly)"
            )

    pgo = pio = _MissingPlotly()
from scotty.geometry import (
    InterpolatedField_Cartesian,
    MagneticField_Cartesian,
    MagneticField_Cylindrical,
)
from scotty.typing import FloatArray, PathLike
from typing import Union, Tuple, Optional, cast

##################################################
#
# matplotlib, for an output DataTree in either geometry
#
##################################################


def maybe_make_axis(ax: Optional[plt.Axes], *args, **kwargs) -> plt.Axes:
    """``ax``, or the axes of a new figure if it is ``None``"""
    if ax is None:
        _, ax = plt.subplots(*args, **kwargs)
    return ax


def _is_cartesian(dt: xr.DataTree) -> bool:
    return str(dt["inputs"]["geometry"].values) == "cartesian"


def beam_path_cylindrical(dt: xr.DataTree) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """The central ray as (R, zeta, Z), from the solver output in either
    geometry: for ``geometry="cartesian"``, R = sqrt(X^2 + Y^2) and
    zeta = atan2(Y, X)"""
    so = dt["solver_output"]
    if _is_cartesian(dt):
        X, Y = so["q_X_output"].values, so["q_Y_output"].values
        return np.hypot(X, Y), np.arctan2(Y, X), so["q_Z_output"].values
    return (
        so["q_R_output"].values,
        so["q_zeta_output"].values,
        so["q_Z_output"].values,
    )


def beam_path_cartesian(dt: xr.DataTree) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """The central ray as (X, Y, Z), from the solver output in either
    geometry"""
    if _is_cartesian(dt):
        so = dt["solver_output"]
        return so["q_X_output"].values, so["q_Y_output"].values, so["q_Z_output"].values
    R, zeta, Z = beam_path_cylindrical(dt)
    return R * np.cos(zeta), R * np.sin(zeta), Z


def _launch_position(dt: xr.DataTree) -> Tuple[FloatArray, FloatArray]:
    """The launch position, (R, zeta, Z) and (X, Y, Z)"""
    so = dt["solver_output"]
    return so["q_launch_cylindrical"].values, so["q_launch_cartesian"].values


def _beam_edges(
    dt: xr.DataTree, orthogonal_dirs: FloatArray
) -> Optional[Tuple[FloatArray, FloatArray]]:
    """The two edges of the beam, q +- w (Cartesian, (N, 3)), where w is the
    beam's half-width vector along g x orthogonal_dir: |w| = sqrt(2 / Im(u .
    Psi . u)) with u its unit vector. ``None`` for a ray-tracing run, which
    has no Psi"""
    so = dt["solver_output"]
    if "Psi_3D_output_labframe_cartesian" not in so:
        return None
    q = np.stack(beam_path_cartesian(dt), axis=-1)
    g_hat = dt["analysis"]["g_hat_cartesian"].values
    Psi = so["Psi_3D_output_labframe_cartesian"].values
    W_vec = np.cross(g_hat, orthogonal_dirs)
    W_hat = W_vec / np.linalg.norm(W_vec, axis=-1, keepdims=True)
    width = np.sqrt(2 / np.einsum("ni,nij,nj->n", W_hat, Psi, W_hat).imag)
    w = W_hat * width[:, np.newaxis]
    return q + w, q - w


def plot_bounding_box(dt: xr.DataTree, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot the (R, Z) bounding box of the magnetic field's grid, from the
    output's inputs. For ``geometry="cartesian"``, the box spans the R of the
    grid's (X, Y) extent"""
    ax = maybe_make_axis(ax)
    inputs = dt["inputs"]
    Z = inputs["Z"].values
    if _is_cartesian(dt):
        X, Y = inputs["X"].values, inputs["Y"].values
        corners = np.hypot(*np.meshgrid([X.min(), X.max()], [Y.min(), Y.max()]))
        contains_axis = X.min() <= 0 <= X.max() and Y.min() <= 0 <= Y.max()
        R_min = 0.0 if contains_axis else float(corners.min())
        R_max = float(corners.max())
    else:
        R = inputs["R"].values
        R_min, R_max = float(R.min()), float(R.max())
    Z_min, Z_max = float(Z.min()), float(Z.max())

    ax.hlines([Z_min, Z_max], R_min, R_max, color="lightgrey", linestyle="dashed")
    ax.vlines([R_min, R_max], Z_min, Z_max, color="lightgrey", linestyle="dashed")
    ax.axis("equal")
    return ax


def plot_flux_surfaces(
    field: Union[MagneticField_Cylindrical, MagneticField_Cartesian],
    ax: Optional[plt.Axes] = None,
    zeta: float = 0.0,
    levels: Optional[FloatArray] = None,
    highlight_LCFS: bool = True,
    n_points: int = 201,
) -> plt.Axes:
    """Contours of the normalised poloidal flux in the (R, Z) plane at the
    toroidal angle ``zeta`` (used by Cartesian fields only), from 0 to 1 in
    steps of 0.1 by default, with the last closed flux surface dashed"""
    ax = maybe_make_axis(ax)
    if isinstance(field, MagneticField_Cylindrical):
        R = np.asarray(field.R_coord)
        Z = np.asarray(field.Z_coord)
        RR, ZZ = np.meshgrid(R, Z, indexing="ij")
        psi = field.polflux(RR, zeta, ZZ)
    else:
        X, Y = np.asarray(field.X_coord), np.asarray(field.Y_coord)
        corners = np.hypot(*np.meshgrid([X.min(), X.max()], [Y.min(), Y.max()]))
        R = np.linspace(0, corners.max(), n_points)
        Z = np.asarray(field.Z_coord)
        RR, ZZ = np.meshgrid(R, Z, indexing="ij")
        psi = field.polflux(RR * np.cos(zeta), RR * np.sin(zeta), ZZ)

    if levels is None:
        levels = np.linspace(0, 1, 11, endpoint=not highlight_LCFS)
    contours = ax.contour(R, Z, np.asarray(psi).T, levels=levels, cmap="plasma_r")
    ax.clabel(contours, inline=True)
    if highlight_LCFS:
        ax.contour(
            R,
            Z,
            np.asarray(psi).T,
            levels=[1.0],
            colors="darkgrey",
            linestyles="dashed",
        )
    ax.axis("equal")
    return ax


def plot_dispersion_relation(
    analysis: Union[xr.Dataset, xr.DataTree],
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot |H| of the three eigenvalues of the dispersion tensor, from
    `numpy.linalg` and from Cardano's formula, along the beam path (distance
    from the cut-off). One of them should be zero all along the path; this
    shows if the root that is zero changes.

    ``analysis`` is the output's ``analysis`` group (``dt["analysis"]``)."""
    ax = maybe_make_axis(ax)
    l_lc = analysis["beam_path_length_wrt_cutoff"].values

    styles = [("pink", "lightgreen", "gray"), ("k", "r", "g")]
    for (name, linestyle), colours in zip(
        (("H_eigvals", "-"), ("H_Cardano", ":")), styles
    ):
        values = np.abs(np.asarray(analysis[name].values))
        for i, colour in enumerate(colours):
            ax.plot(
                l_lc,
                values[:, i],
                color=colour,
                linestyle=linestyle,
                linewidth=4,
                label=f"{name}, solution {i + 1}",
            )

    ax.set_title("Dispersion relation")
    ax.set_xlabel("Distance along the beam from the cut-off [m]")
    ax.set_ylabel("|H|")
    ax.legend()

    if filename:
        plt.savefig(filename)
    return ax


def plot_poloidal_beam_path(
    dt: xr.DataTree,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
    zoom: bool = False,
    field: Optional[Union[MagneticField_Cylindrical, MagneticField_Cartesian]] = None,
) -> plt.Axes:
    """Plot the beam path in the (R, Z) plane: the central ray, from the
    launch position (in vacuum, dotted) through the solver's points, the beam
    width in that plane (dashed), and the field's bounding box. With
    ``field``, also its flux surfaces. With ``zoom``, the axes are limited to
    the beam."""
    ax = maybe_make_axis(ax)

    R, zeta, Z = beam_path_cylindrical(dt)
    if field is not None:
        plot_flux_surfaces(field, ax=ax, zeta=float(np.mean(zeta)))
    plot_bounding_box(dt, ax)

    (launch_R, _, launch_Z), _ = _launch_position(dt)
    ax.plot(
        [launch_R, R[0]],
        [launch_Z, Z[0]],
        ":k",
        label="Central (reference) ray in vacuum",
    )
    ax.plot(R, Z, "-k", label="Central (reference) ray")

    # Width in the poloidal plane: perpendicular to g and to the toroidal
    # direction at each point
    toroidal_dirs = np.stack([-np.sin(zeta), np.cos(zeta), np.zeros_like(zeta)], -1)
    edges = _beam_edges(dt, toroidal_dirs)
    if edges is not None:
        for i, edge in enumerate(edges):
            ax.plot(
                np.hypot(edge[:, 0], edge[:, 1]),
                edge[:, 2],
                "--k",
                label="Beam width" if i else None,
            )
    ax.scatter(launch_R, launch_Z, c="red", marker=">", label="Launch position")

    if zoom:
        points = [np.append(R, launch_R), np.append(Z, launch_Z)]
        if edges is not None:
            for edge in edges:
                points[0] = np.append(points[0], np.hypot(edge[:, 0], edge[:, 1]))
                points[1] = np.append(points[1], edge[:, 2])
        for limits, values in zip((ax.set_xlim, ax.set_ylim), points):
            low, high = np.nanmin(values), np.nanmax(values)
            buffer = 0.1 * (high - low)
            limits(low - buffer, high + buffer)

    ax.legend()
    ax.set_title("Beam path (poloidal plane)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")

    if filename:
        plt.savefig(filename)
    return ax


def plot_toroidal_beam_path(
    dt: xr.DataTree,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
    field: Optional[Union[MagneticField_Cylindrical, MagneticField_Cartesian]] = None,
) -> plt.Axes:
    """Plot the beam path seen from above, in the (X, Y) plane: the central
    ray from the launch position, and the beam width in that plane (dashed).
    With ``field``, also the last closed flux surface on the midplane
    (Z = 0)."""
    ax = maybe_make_axis(ax)

    if field is not None:
        if isinstance(field, MagneticField_Cylindrical):
            R_grid = np.asarray(field.R_coord)
            psi = field.polflux(R_grid, 0.0, np.zeros_like(R_grid)) - 1.0
            crossings = np.nonzero(np.diff(np.sign(psi)))[0]
            angle = np.linspace(-np.pi, np.pi, 1001)
            for i in crossings:
                # Linear interpolation between the two grid points
                R_lcfs = R_grid[i] - psi[i] * (R_grid[i + 1] - R_grid[i]) / (
                    psi[i + 1] - psi[i]
                )
                ax.plot(R_lcfs * np.cos(angle), R_lcfs * np.sin(angle), "orange")
        else:
            X, Y = np.asarray(field.X_coord), np.asarray(field.Y_coord)
            XX, YY = np.meshgrid(X, Y, indexing="ij")
            psi = field.polflux(XX, YY, np.zeros_like(XX))
            ax.contour(X, Y, np.asarray(psi).T, levels=[1.0], colors="orange")

    X, Y, _ = beam_path_cartesian(dt)
    _, (launch_X, launch_Y, _) = _launch_position(dt)
    ax.plot(
        np.concatenate([[launch_X], X]),
        np.concatenate([[launch_Y], Y]),
        ":k",
        label="Central (reference) ray",
    )

    # Width in the (X, Y) plane: perpendicular to g and to Z
    edges = _beam_edges(dt, np.array([0.0, 0.0, 1.0]))
    if edges is not None:
        for i, edge in enumerate(edges):
            ax.plot(edge[:, 0], edge[:, 1], "--k", label="Beam width" if i else None)
    ax.scatter(launch_X, launch_Y, c="red", marker=">", label="Launch position")

    ax.legend()
    ax.set_title("Beam path (toroidal plane)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")

    if filename:
        plt.savefig(filename)
    return ax


##################################################
#
# plotly, in 3D
#
##################################################


def make_fig_if_None(existing_fig: Optional[pgo.Figure] = None) -> pgo.Figure:
    fig = pgo.Figure() if existing_fig is None else existing_fig
    return fig


def plot_3d_flux_surfaces(
    field: Union[MagneticField_Cartesian, MagneticField_Cylindrical],
    xlims: Tuple[float, float, float] = (0, 2.5, 101),
    ylims: Tuple[float, float, float] = (-0.5, 0.5, 41),
    zlims: Tuple[float, float, float] = (-1.5, 1.5, 121),
    existing_fig: Optional[pgo.Figure] = None,
    use_rho: bool = False,
    show_graph: bool = False,
    show_in_browser: bool = False,
    show_axes: bool = True,
) -> pgo.Figure:

    Z = field.Z_coord if zlims is None else np.linspace(*zlims)  # type: ignore

    if isinstance(field, (MagneticField_Cartesian, InterpolatedField_Cartesian)):
        X = field.X_coord if xlims is None else np.linspace(*xlims)  # type: ignore
        Y = field.Y_coord if ylims is None else np.linspace(*ylims)  # type: ignore
        XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing="ij")
        flux_coord = (
            np.sqrt(field.polflux_cartesian(XX, YY, ZZ))
            if use_rho
            else field.polflux_cartesian(XX, YY, ZZ)
        )
        flux_coord = cast(FloatArray, flux_coord)

    else:  # elif isinstance(field, MagneticField_Cylindrical, InterpolatedField_Cylindrical):
        X = field.R_coord if xlims is None else np.linspace(*xlims)  # type: ignore
        Y = np.linspace(-0.5, 0.5, 41) if ylims is None else np.linspace(*ylims)  # type: ignore
        XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing="ij")
        flux_coord = (
            np.sqrt(field.poloidal_flux(np.sqrt(XX**2 + YY**2), ZZ))
            if use_rho
            else field.poloidal_flux(np.sqrt(XX**2 + YY**2), ZZ)
        )
        flux_coord = cast(FloatArray, flux_coord)

    # masking to remove NaNs and outside LCFS, to see if we can speed up plotting time
    # flux_threshold = 1.0 + 0.1 # leave this here just in case we want to implement this in the future
    mask = np.isfinite(
        flux_coord
    )  # & (flux_coord <= flux_threshold) # masking like this prevents plotly from plotting, see if can fix
    XX = XX[mask]
    YY = YY[mask]
    ZZ = ZZ[mask]
    flux_coord = flux_coord[mask]

    fig = make_fig_if_None(existing_fig)

    # Built outside the f-string: before Python 3.12 its {} can't hold a backslash
    flux_label = r"$\rho_n$" if use_rho else r"$\psi_n$"

    fig.add_trace(
        pgo.Isosurface(
            x=XX.ravel(),
            y=YY.ravel(),
            z=ZZ.ravel(),
            value=flux_coord.ravel(),
            isomin=0.0,
            isomax=1.0,
            surface_count=11,
            opacity=0.1,
            colorscale="plasma_r",
            caps={"x_show": show_axes, "y_show": show_axes, "z_show": show_axes},
            name=f"Flux surfaces, {flux_label}",
            colorbar={
                "title": flux_label,
                "thickness": 10,
                "len": 0.7,
            },
        )
    )

    fig.update_layout(
        title="Testing",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            xaxis=dict(range=xlims[:2]),
            yaxis=dict(range=ylims[:2]),
            zaxis=dict(range=zlims[:2]),
            aspectmode="data",
        ),
        width=900,
        height=700,
    )

    if show_graph:
        if show_in_browser:
            pio.renderers.default = "browser"
        fig.update_layout(showlegend=True)
        fig.show()

    return fig


def plot_3d_beam(
    q_X: FloatArray,
    q_Y: FloatArray,
    q_Z: FloatArray,
    widths_N: FloatArray,
    widths_B: FloatArray,
    n_hat: FloatArray,
    b_hat: FloatArray,
    theta_resolution: int = 60,
    existing_fig: Optional[pgo.Figure] = None,
    show_graph: bool = True,
    show_in_browser: bool = True,
) -> pgo.Figure:
    """
    Vectorised 3D beam visualization using Plotly Mesh3d

    field : MagneticField_Cartesian or MagneticField_Cylindrical
        Magnetic field class
    q_X, q_Y, q_Z : (N,) FloatArray
        Central ray coordinates
    widths_N, widths_B : (N,) FloatArray
        Beam radii along N(ormal) and B(inormal) directions
    n_hat, b_hat : (N,3) FloatArray
        N(ormal) and B(inormal) unit vectors
    theta_resolution : int
        Angular resolution in the beam cross-section
    xlims, ylims, zlims : Tuple
        Three element tuple containing (`min`, `max`, `num_step`)
    existing_fig : pgo.Figure or None
        If `pgo.Figure`, add the trace there; otherwise, create a new plotly figure
    show_graph : bool
        If True, display the graph
    show_in_browser : bool
        If True, then the graph is displayed in the browser
    """

    # Vectorise the beam silhouette (beam_surf)
    q_XX = q_X[:, None]  # shape (N,1)
    q_YY = q_Y[:, None]  # shape (N,1)
    q_ZZ = q_Z[:, None]  # shape (N,1)
    w_NN = widths_N[:, None]  # shape (N,1)
    w_BB = widths_B[:, None]  # shape (N,1)

    n_T = n_hat.T  # shape (3,N)
    b_T = b_hat.T  # shape (3,N)

    theta = np.linspace(0, 2 * np.pi, theta_resolution)
    sin_theta = np.sin(theta)[None, :]  # shape (1,T)
    cos_theta = np.cos(theta)[None, :]  # shape (1,T)

    X_beam_surf = (
        q_XX
        + (w_NN * cos_theta * n_T[0][:, None])
        + (w_BB * sin_theta * b_T[0][:, None])
    ).ravel()
    Y_beam_surf = (
        q_YY
        + (w_NN * cos_theta * n_T[1][:, None])
        + (w_BB * sin_theta * b_T[1][:, None])
    ).ravel()
    Z_beam_surf = (
        q_ZZ
        + (w_NN * cos_theta * n_T[2][:, None])
        + (w_BB * sin_theta * b_T[2][:, None])
    ).ravel()

    # Get triangular mesh indices for the corresponding points
    # of the beam silhouette, where the vertices are connected like
    # triangle set 1: (i, j) -> (i, j+1) -> (i+1, j+1)
    # triangle set 2: (i, j) -> (i+1, j+1) -> (i+1, j)
    # and stack them all together
    ii = np.arange(len(q_X))
    jj = np.arange(theta_resolution)
    I, J = np.meshgrid(ii[:-1], jj[:-1], indexing="ij")
    I, J = I.ravel(), J.ravel()

    def tri_map(i, j):
        return i * theta_resolution + j

    t_i = np.concatenate([tri_map(I, J), tri_map(I, J)])
    t_j = np.concatenate([tri_map(I, J + 1), tri_map(I + 1, J + 1)])
    t_k = np.concatenate([tri_map(I + 1, J + 1), tri_map(I + 1, J)])

    # Some plotly stuff
    fig = make_fig_if_None(existing_fig)

    # Plot central ray
    fig.add_trace(
        pgo.Scatter3d(
            x=q_X,
            y=q_Y,
            z=q_Z,
            mode="lines",
            line=dict(color="black", width=6),
            name="Central ray (plasma)",
        )
    )

    # Plot beam surface
    fig.add_trace(
        pgo.Mesh3d(
            x=X_beam_surf,
            y=Y_beam_surf,
            z=Z_beam_surf,
            i=t_i,
            j=t_j,
            k=t_k,
            color="black",
            opacity=0.25,
            name="1/e Beam (plasma)",
        )
    )

    if show_graph:
        if show_in_browser:
            pio.renderers.default = "browser"
        fig.update_layout(showlegend=True)
        fig.show()

    return fig


def plot_3d_beam_in_plasma(
    field: Union[MagneticField_Cartesian, MagneticField_Cylindrical],
    q_X: FloatArray,
    q_Y: FloatArray,
    q_Z: FloatArray,
    widths_N: FloatArray,
    widths_B: FloatArray,
    n_hat: FloatArray,
    b_hat: FloatArray,
    theta_resolution: int = 60,
    xlims: Tuple[float, float, float] = (0, 2.5, 101),
    ylims: Tuple[float, float, float] = (-0.5, 0.5, 41),
    zlims: Tuple[float, float, float] = (-1.5, 1.5, 121),
    existing_fig: Optional[pgo.Figure] = None,
    use_rho: bool = False,
    show_graph: bool = True,
    show_in_browser: bool = True,
    show_axes: bool = True,
) -> pgo.Figure:

    fig = make_fig_if_None(existing_fig)

    fig = plot_3d_flux_surfaces(
        field=field,
        xlims=xlims,
        ylims=ylims,
        zlims=zlims,
        existing_fig=fig,
        use_rho=use_rho,
        show_axes=show_axes,
        show_graph=False,
        show_in_browser=False,
    )

    fig = plot_3d_beam(
        q_X=q_X,
        q_Y=q_Y,
        q_Z=q_Z,
        widths_N=widths_N,
        widths_B=widths_B,
        n_hat=n_hat,
        b_hat=b_hat,
        theta_resolution=theta_resolution,
        existing_fig=fig,
        show_graph=show_graph,
        show_in_browser=show_in_browser,
    )

    return fig
