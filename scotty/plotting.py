from contourpy import contour_generator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import xarray as xr
from datatree import DataTree
from scipy.interpolate import CubicSpline

from .analysis import beam_width
from .geometry import MagneticField
from .fun_general import cylindrical_to_cartesian
from .launch import find_entry_point

from typing import Optional, Union, List
from scotty.typing import FloatArray, PathLike


def maybe_make_axis(ax: Optional[plt.Axes], *args, **kwargs) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(*args, **kwargs)
    return ax


def maybe_make_3D_axis(ax: Optional[plt.Axes], *args, **kwargs) -> plt.Axes:
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    return ax


def plot_bounding_box(field: MagneticField, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot the bounding box of the magnetic field1

    Parameters
    ----------
    field : MagneticField
        Magnetic field object
    ax : Optional[plt.Axes]
        Axis to add bounding box to. If ``None``, creates a new figure
        and axes

    Returns
    -------
    plt.Axes
        Axes bounding box was added to

    """

    ax = maybe_make_axis(ax)

    R_min = np.min(field.R_coord)
    R_max = np.max(field.R_coord)
    Z_min = np.min(field.Z_coord)
    Z_max = np.max(field.Z_coord)

    ax.hlines([Z_min, Z_max], R_min, R_max, color="lightgrey", linestyle="dashed")
    ax.vlines([R_min, R_max], Z_min, Z_max, color="lightgrey", linestyle="dashed")
    bounds = np.array((R_min, R_max, Z_min, Z_max))
    ax.axis(bounds * 1.1)
    ax.axis("equal")

    return ax


def plot_bounding_box_3D(
    field: MagneticField, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot the bounding box of the magnetic field1

    Parameters
    ----------
    field : MagneticField
        Magnetic field object
    ax : Optional[plt.Axes]
        Axis to add bounding box to. If ``None``, creates a new figure
        and axes

    Returns
    -------
    plt.Axes
        Axes bounding box was added to

    """
    ax = maybe_make_3D_axis(ax)

    R_max = np.max(field.R_coord)
    Z_min = np.min(field.Z_coord)
    Z_max = np.max(field.Z_coord)

    # horizontal lines
    for Z in (Z_min, Z_max):
        ax.plot((-R_max, R_max), (-R_max, -R_max), (Z, Z), "tab:blue")
        ax.plot((-R_max, R_max), (R_max, R_max), (Z, Z), "tab:blue")
        ax.plot((-R_max, -R_max), (-R_max, R_max), (Z, Z), "tab:blue")
        ax.plot((R_max, R_max), (-R_max, R_max), (Z, Z), "tab:blue")

    # vertical lines
    ax.plot((-R_max, -R_max), (-R_max, -R_max), (Z_min, Z_max), "tab:blue")
    ax.plot((R_max, R_max), (-R_max, -R_max), (Z_min, Z_max), "tab:blue")
    ax.plot((R_max, R_max), (R_max, R_max), (Z_min, Z_max), "tab:blue")
    ax.plot((-R_max, -R_max), (R_max, R_max), (Z_min, Z_max), "tab:blue")

    ax.axis("equal")

    return ax


def plot_flux_surface(
    field: MagneticField,
    surface: Union[float, List[float]],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot one or more flux surfaces


    Parameters
    ----------
    field : MagneticField
        Magnetic field object
    surface : Union[float, List[float]]
        If ``surface`` is a single ``float``, then plot just that
        surface as a dashed black line. Otherwise, plot all the
        surfaces and add a colourbar to the plot.
    ax : Optional[plt.Axes]
        Axis to plot to. If ``None``, creates a new figure and axes

    Returns
    -------
    plt.Axes
        Axes flux surfaces were added to

    """

    ax = maybe_make_axis(ax)

    style = {}
    if isinstance(surface, float):
        style = {"colors": "black", "linestyles": "dashed"}
        surface = [surface]

    ax.contour(
        field.R_coord, field.Z_coord, field.poloidalFlux_grid.T, levels=surface, **style
    )
    ax.axis("equal")

    return ax


def plot_poloidal_crosssection(
    field: MagneticField, ax: Optional[plt.Axes] = None, highlight_LCFS: bool = True
) -> plt.Axes:
    ax = maybe_make_axis(ax)

    plot_bounding_box(field, ax)

    cax = ax.contour(
        field.R_coord,
        field.Z_coord,
        field.poloidalFlux_grid.T,
        levels=np.linspace(0, 1, 10, endpoint=not highlight_LCFS),
        cmap="plasma_r",
    )
    ax.clabel(cax, inline=True)
    if highlight_LCFS:
        ax.contour(
            field.R_coord,
            field.Z_coord,
            field.poloidalFlux_grid.T,
            levels=[1.0],
            colors="darkgrey",
            linestyles="dashed",
        )
        dashed_line = mlines.Line2D(
            [], [], color="black", linestyle="dashed", label=r"$\psi = 1$"
        )
        ax.legend(handles=[dashed_line])

    return ax


def plot_flux_surface_3D(
    field: MagneticField, psi: float, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    contour = contour_generator(
        x=field.R_coord, y=field.Z_coord, z=field.poloidalFlux_grid.T
    )

    ax = maybe_make_3D_axis(ax)
    R, Z = np.array(contour.lines(psi)[0]).T
    phi = np.linspace(0, 2 * np.pi)

    R_, phi_ = np.meshgrid(R, phi, indexing="ij")
    X = R_ * np.cos(phi_)
    Y = R_ * np.sin(phi_)

    ax.plot_surface(
        X, Y, np.broadcast_to(Z[:, np.newaxis], X.shape), antialiased=True, alpha=0.75
    )
    ax.axis("equal")
    return ax


def plot_all_the_things(
    field: MagneticField,
    launch_position: FloatArray,
    poloidal_launch_angle: float,
    toroidal_launch_angle: float,
    poloidal_flux_enter: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    ax = maybe_make_3D_axis(ax)
    plot_flux_surface_3D(field, 1.0, ax)
    plot_bounding_box_3D(field, ax)

    entry_point = cylindrical_to_cartesian(
        *find_entry_point(
            launch_position,
            poloidal_launch_angle,
            toroidal_launch_angle,
            poloidal_flux_enter,
            field,
        )
    )

    ax.plot(
        *launch_position, marker="x", markersize=5, markeredgewidth=4, color="tab:red"
    )
    ax.plot(*zip(launch_position, entry_point), color="tab:red")

    return ax


def plot_dispersion_relation(
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots Cardano's and np.linalg's solutions to the actual dispersion relation
    Useful to check whether the solution which = 0 along the path changes
    """

    ax = maybe_make_axis(ax)

    np.abs(analysis.H_eigvals).plot(x="l_lc", hue="col", ax=ax)

    ax.plot(
        analysis.l_lc, abs(analysis.H_1_Cardano), linestyle="--", label="H_1_Cardano"
    )
    ax.plot(
        analysis.l_lc, abs(analysis.H_2_Cardano), linestyle="--", label="H_2_Cardano"
    )
    ax.plot(
        analysis.l_lc, abs(analysis.H_3_Cardano), linestyle="--", label="H_3_Cardano"
    )
    ax.set_title("Dispersion relation")

    if filename:
        plt.savefig(f"{filename}.png")

    return ax


def plot_poloidal_beam_path(
    dt: DataTree,
    field: MagneticField,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plots the beam path on the R Z plane

    .. todo:: Would be nice to not have to pass in ``field``
    """

    ax = maybe_make_axis(ax)

    plot_poloidal_crosssection(field, ax=ax, highlight_LCFS=False)

    launch_R = dt.inputs.launch_position.sel(col="R")
    launch_Z = dt.inputs.launch_position.sel(col="Z")
    ax.plot(
        np.concatenate([[launch_R], dt.analysis.q_R]),
        np.concatenate([[launch_Z], dt.analysis.q_Z]),
        ":k",
        label="Central (reference) ray",
    )
    width = beam_width(dt.analysis.g_hat, np.array([0.0, 1.0, 0.0]), dt.analysis.Psi_3D)
    beam_plus = dt.analysis.beam + width
    beam_minus = dt.analysis.beam - width
    ax.plot(beam_plus.sel(col="R"), beam_plus.sel(col="Z"), "--k")
    ax.plot(beam_minus.sel(col="R"), beam_minus.sel(col="Z"), "--k", label="Beam width")
    ax.scatter(launch_R, launch_Z, c="red", marker=">", label="Launch position")

    ax.legend()
    ax.set_title("Beam path (poloidal plane)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")

    if filename:
        plt.savefig(f"{filename}.png")

    return ax


def plot_toroidal_contour(ax: plt.Axes, R: float, zeta: FloatArray, colour="orange"):
    x, y, _ = cylindrical_to_cartesian(
        R * np.ones_like(zeta), zeta, np.zeros_like(zeta)
    )
    ax.plot(x, y, colour)


def plot_toroidal_beam_path(
    dt: DataTree, filename: Optional[PathLike] = None, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    ax = maybe_make_axis(ax)

    lcfs = 1
    flux_spline = CubicSpline(
        dt.analysis.R_midplane, dt.analysis.poloidal_flux_on_midplane - lcfs
    )
    all_R_lcfs = flux_spline.roots()

    zeta = np.linspace(-np.pi, np.pi, 1001)
    for R_lcfs in all_R_lcfs:
        plot_toroidal_contour(ax, R_lcfs, zeta)

    flux_min_index = dt.analysis.poloidal_flux_on_midplane.argmin()
    R_axis = dt.analysis.R_midplane[flux_min_index].data
    plot_toroidal_contour(ax, R_axis, zeta, "#00003f")

    launch_X, launch_Y, _ = cylindrical_to_cartesian(*dt.inputs.launch_position)
    ax.plot(
        np.concatenate([[launch_X], dt.analysis.q_X]),
        np.concatenate([[launch_Y], dt.analysis.q_Y]),
        ":k",
        label="Central (reference) ray",
    )
    width = beam_width(
        dt.analysis.g_hat_Cartesian,
        np.array([0.0, 0.0, 1.0]),
        dt.analysis.Psi_3D_Cartesian,
    )
    beam_plus = dt.analysis.beam_cartesian + width
    beam_minus = dt.analysis.beam_cartesian - width
    ax.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Y"), "--k")
    ax.plot(
        beam_minus.sel(col_cart="X"),
        beam_minus.sel(col_cart="Y"),
        "--k",
        label="Beam width",
    )
    ax.scatter(launch_X, launch_Y, c="red", marker=">", label="Launch position")

    ax.legend()
    ax.set_title("Beam path (toroidal plane)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    if filename:
        plt.savefig(f"{filename}.png")

    return ax
