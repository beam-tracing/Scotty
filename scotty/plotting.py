from contourpy import contour_generator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from .geometry import MagneticField
from .fun_general import cylindrical_to_cartesian
from .launch import find_entry_point

from typing import Optional, Union, List
from scotty.typing import FloatArray


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

    if ax is None:
        _, ax = plt.subplots()

    R_min = np.min(field.R_coord)
    R_max = np.max(field.R_coord)
    Z_min = np.min(field.Z_coord)
    Z_max = np.max(field.Z_coord)

    ax.hlines([Z_min, Z_max], R_min, R_max)
    ax.vlines([R_min, R_max], Z_min, Z_max)
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

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

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

    if ax is None:
        _, ax = plt.subplots()

    style = {}
    if isinstance(surface, float):
        style = {"colors": "black", "linestyles": "dashed"}
        surface = [surface]

    ax.contour(
        field.R_coord, field.Z_coord, field.poloidalFlux_grid.T, levels=surface, **style
    )
    ax.axis("equal")

    return ax


def plot_field(field: MagneticField):
    ax = plot_bounding_box(field)

    cax = ax.contour(
        field.R_coord,
        field.Z_coord,
        field.poloidalFlux_grid.T,
        levels=np.linspace(0, 1, 10, endpoint=False),
    )
    ax.clabel(cax, inline=True)
    ax.contour(
        field.R_coord,
        field.Z_coord,
        field.poloidalFlux_grid.T,
        levels=[1.0],
        colors="black",
        linestyles="dashed",
        linewidths=2,
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

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

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
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

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
