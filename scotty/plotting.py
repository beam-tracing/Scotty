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


from matplotlib.widgets import Slider
from scotty.fun_general import find_q_lab_Cartesian
import datatree


def maybe_make_axis(ax: Optional[plt.Axes], *args, **kwargs) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(*args, **kwargs)
    return ax


def maybe_make_3D_axis(ax: Optional[plt.Axes], *args, **kwargs) -> plt.Axes:
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    return ax


def plot_bounding_box(dt: DataTree, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot the bounding box of the magnetic field

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

    R_min = np.min(dt["inputs"].R.values)
    R_max = np.max(dt["inputs"].R.values)
    Z_min = np.min(dt["inputs"].Z.values)
    Z_max = np.max(dt["inputs"].Z.values)

    ax.hlines([Z_min, Z_max], R_min, R_max, color="lightgrey", linestyle="dashed")
    ax.vlines([R_min, R_max], Z_min, Z_max, color="lightgrey", linestyle="dashed")
    bounds = np.array((R_min, R_max, Z_min, Z_max))
    ax.axis(bounds * 1.1)
    ax.axis("equal")

    return ax


def plot_bounding_box_3D(
    field: MagneticField, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot the bounding box of the magnetic field

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
    dt: DataTree, ax: Optional[plt.Axes] = None, highlight_LCFS: bool = True
) -> plt.Axes:
    """
    Plots the poloidal cross section
    """
    ax = maybe_make_axis(ax)

    plot_bounding_box(dt, ax)

    cax = ax.contour(
        dt["inputs"].R.values,
        dt["inputs"].Z.values,
        (dt["inputs"].poloidalFlux_grid.values).T,
        levels=np.linspace(0, 1, 11, endpoint=not highlight_LCFS),
        cmap="plasma_r",
    )
    ax.clabel(cax, inline=True)
    if highlight_LCFS:
        ax.contour(
            dt["inputs"].R.values,
            dt["inputs"].Z.values,
            (dt["inputs"].poloidalFlux_grid.values).T,
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

    ## The title of 'col' in H_eigvals is a misnomer
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_eigvals.sel(col="R")),
        color="pink",
        linestyle="-",
        linewidth=4,
        label="H_eigvals_1 (X?)",
    )
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_eigvals.sel(col="zeta")),
        color="lightgreen",
        linestyle="-",
        linewidth=4,
        label="H_eigvals_2 (O?)",
    )
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_eigvals.sel(col="Z")),
        color="gray",
        linestyle="-",
        linewidth=4,
        label="H_eigvals_3",
    )

    ax.plot(
        analysis.l_lc,
        abs(analysis.H_1_Cardano),
        color="k",
        linestyle=":",
        linewidth=4,
        label="H_1_Cardano",
    )
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_2_Cardano),
        color="r",
        linestyle=":",
        linewidth=4,
        label="H_2_Cardano (X?)",
    )
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_3_Cardano),
        color="g",
        linestyle=":",
        linewidth=4,
        label="H_3_Cardano (O?)",
    )

    ax.set_title("Dispersion relation")
    ax.legend()

    if filename:
        plt.savefig(f"{filename}.png")

    return ax


def plot_poloidal_beam_path(
    dt: DataTree,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
    zoom=False,
) -> plt.Axes:
    """
    Plots the beam path on the R Z plane (poloidal cross section)
    """

    ax = maybe_make_axis(ax)

    plot_poloidal_crosssection(dt, ax=ax, highlight_LCFS=False)

    launch_R = dt.inputs.launch_position.sel(col="R")
    launch_Z = dt.inputs.launch_position.sel(col="Z")
    # plot piece in vacuum separately
    ax.plot(
        [launch_R, dt.analysis.q_R[0]],
        [launch_Z, dt.analysis.q_Z[0]],
        ":k",
        label="Central (reference) ray in vacuum",
    )

    # now plot piece in plasma
    ax.plot(
        dt.analysis.q_R,
        dt.analysis.q_Z,
        "-k",
        label="Central (reference) ray in plasma",
    )

    width = beam_width(dt.analysis.g_hat, np.array([0.0, 1.0, 0.0]), dt.analysis.Psi_3D)
    beam_plus = dt.analysis.beam + width
    beam_minus = dt.analysis.beam - width
    ax.plot(beam_plus.sel(col="R"), beam_plus.sel(col="Z"), "--k")
    ax.plot(beam_minus.sel(col="R"), beam_minus.sel(col="Z"), "--k", label="Beam width")
    ax.scatter(launch_R, launch_Z, c="red", marker=">", label="Launch position")

    if zoom:
        ## Write a wrapper function for this maybe
        R_max = max(beam_plus.sel(col="R").max(), beam_minus.sel(col="R").max())
        R_min = min(beam_plus.sel(col="R").min(), beam_minus.sel(col="R").min())
        Z_max = max(beam_plus.sel(col="Z").max(), beam_minus.sel(col="Z").max())
        Z_min = min(beam_plus.sel(col="Z").min(), beam_minus.sel(col="Z").min())

        buffer_R = 0.1 * (R_max - R_min)
        buffer_Z = 0.1 * (Z_max - Z_min)

        ax.set_xlim(R_min - buffer_R, R_max + buffer_R)
        ax.set_ylim(Z_min - buffer_Z, Z_max + buffer_Z)

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
    """
    Plots the trajectory of the beam in the toroidal plane
    """
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


def plot_widths(
    dt: DataTree, filename: Optional[PathLike] = None, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots a graph of the beam width vs the distance along the ray
    """
    ax = maybe_make_axis(ax)

    Psi_w_imag = np.array(
        np.imag(
            [
                [dt["analysis"].Psi_xx.values, dt["analysis"].Psi_xy.values],
                [dt["analysis"].Psi_xy.values, dt["analysis"].Psi_yy.values],
            ]
        )
    )

    eigvals_imag = np.linalg.eigvalsh(np.moveaxis(Psi_w_imag, -1, 0))
    widths = np.sqrt(2 / eigvals_imag)

    ax.plot(dt["analysis"].distance_along_line.values, widths)
    ax.set_ylabel("widths / m")
    ax.set_xlabel("l / m")

    if filename:
        plt.savefig(f"{filename}.png")

    return ax


def plot_instrumentation_functions(
    dt: DataTree, filename: Optional[PathLike] = None, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots the different localisation terms
    """
    ax = maybe_make_axis(ax, 3, 2)

    x_axis = dt["analysis"].distance_along_line.values

    # ax[0,0].set_title('mismatch')
    # ax[0,0].plot(x_axis,np.rad2deg(dt["analysis"].theta_m.values),label=r'$\theta_m$')
    # ax[0,0].plot(x_axis,np.rad2deg(dt["analysis"].delta_theta_m.values),label=r'$\Delta \theta_m$')
    # ax[0,0].legend()
    # ax[0,0].set_xlabel(r'$l$ / m')
    # ax[0,0].set_ylabel('deg')
    loc_all = (
        # dt["analysis"].loc_b.values
        dt["analysis"].loc_m.values
        * dt["analysis"].loc_p.values
        * dt["analysis"].loc_r.values
    )
    loc_all_and_spectrum = loc_all * dt["analysis"].loc_s.values

    ax[0, 0].plot(
        x_axis, loc_all_and_spectrum, label="loc mprs"
    )  # product of all localisation terms
    ax[0, 0].legend()
    ax[0, 0].set_xlabel(r"$l$ / m")
    ax[0, 0].set_ylabel("deg")

    ax[1, 0].plot(x_axis, dt["analysis"].loc_m.values, label="loc m")  # mismatch term
    ax[1, 0].set_xlabel(r"$l$ / m")
    ax[1, 0].legend()

    ax[2, 0].plot(x_axis, dt["analysis"].loc_b.values, label="loc b")  # beam term
    ax[2, 0].set_xlabel(r"$l$ / m")
    ax[2, 0].legend()

    ax[0, 1].plot(x_axis, dt["analysis"].loc_r.values, label="loc r")  # ray term
    ax[0, 1].set_xlabel(r"$l$ / m")
    ax[0, 1].legend()

    ax[1, 1].plot(x_axis, dt["analysis"].loc_s.values, label="loc s")  # spectrum term
    ax[1, 1].set_xlabel(r"$l$ / m")
    ax[1, 1].legend()

    ax[2, 1].plot(
        x_axis, dt["analysis"].loc_p.values, label="loc p"
    )  # polarisation term
    ax[2, 1].set_xlabel(r"$l$ / m")
    ax[2, 1].legend()

    if filename:
        plt.savefig(f"{filename}.png")

    return ax


def plot_3D_beam_profile_3D_plotting(
    dt: DataTree,
    filename: Optional[PathLike] = None,
    include_p: Optional = False,
    include_s: Optional = False,
    include_b: Optional = False,
    include_r: Optional = False,
    include_m: Optional = False,
):
    """
    Plots a 3D plot of the Beam profile, along with:
    - Eigenvector plot showing the beam width axis
    - Width of the beam along the ray
    - Curvature of the beam along the ray
    - localisations
    """
    # include_...: whether or not to mutliply these values in the localisation graph
    # eg. include_p & include_s True -> plot graph of loc_p * loc_s against distance

    include_width1 = True  # green beam width
    include_width2 = True  # red beam width

    include_title = True
    title_additional_notes = ""
    thresh_gradgrad = 100.0
    thresh_intersect = 100.0
    window_size = 0.01
    ellipse_resolution = 30

    ##extracting inputs
    title_shot = 189998
    title_time = 3005
    title_mode = dt.inputs.mode_flag.data
    title_freq = dt.inputs.launch_freq_GHz.data
    title_pol = dt.inputs.poloidal_launch_angle_Torbeam
    title_tor = dt.inputs.toroidal_launch_angle_Torbeam
    title_width = dt.inputs.launch_beam_width
    title_curv = dt.inputs.launch_beam_curvature

    # For the 3D Axes, will need the limits of the path, with some leeway for the beam
    # note to self, capital coordinates are lab, cylindrical coordinates
    tau_array = dt.analysis.tau
    q_R_array = dt.analysis.q_R
    q_zeta_array = dt.analysis.q_zeta
    q_Z_array = dt.analysis.q_Z
    poloidal_flux_output = dt.analysis.poloidal_flux

    # Here we can calculate the linestep, but importing the array is simpler
    distance_along_line = dt.analysis.distance_along_line.data

    # Here we import the arrays necessary to calculate the beam shape
    Psi_xx_output = dt.analysis.Psi_xx.data
    Psi_xy_output = dt.analysis.Psi_xy.data
    Psi_yy_output = dt.analysis.Psi_yy.data
    x_hat_Cartesian = dt.analysis.x_hat_Cartesian.data
    y_hat_Cartesian = dt.analysis.y_hat_Cartesian.data
    g_hat_output = dt.analysis.g_hat.data
    K_magnitude_array = dt.analysis.K_magnitude.data
    theta_output = dt.analysis.theta.data
    theta_m_output = dt.analysis.theta_m.data
    cutoff_index = dt.analysis.cutoff_index.data

    ##for calculating product of localisations:
    loc_p = dt.analysis.loc_p.data
    loc_s = dt.analysis.loc_s.data
    loc_m = dt.analysis.loc_m.data
    loc_b = dt.analysis.loc_b.data
    loc_r = dt.analysis.loc_r.data

    # ==================================================================================#
    # Code to find the necessary arrays to plot

    [q_X_array, q_Y_array, q_Z_array] = find_q_lab_Cartesian(
        np.array([q_R_array, q_zeta_array, q_Z_array])
    )  # to plot the path in the X,Y,Z space

    numberOfDataPoints = np.size(q_R_array)
    out_index = numberOfDataPoints - 1
    ##convert g_hat to cartesian
    g_hat_Cartesian = np.zeros([numberOfDataPoints, 3])
    g_hat_Cartesian[:, 0] = g_hat_output[:, 0] * np.cos(q_zeta_array) - g_hat_output[
        :, 1
    ] * np.sin(q_zeta_array)
    g_hat_Cartesian[:, 1] = g_hat_output[:, 0] * np.sin(q_zeta_array) + g_hat_output[
        :, 1
    ] * np.cos(q_zeta_array)
    g_hat_Cartesian[:, 2] = g_hat_output[:, 2]
    ##setting up the psi_w arrays, separating real and imaginary parts
    Psi_w_real = np.array(
        np.real([[Psi_xx_output, Psi_xy_output], [Psi_xy_output, Psi_yy_output]])
    )
    Psi_w_imag = np.array(
        np.imag([[Psi_xx_output, Psi_xy_output], [Psi_xy_output, Psi_yy_output]])
    )

    eigvals_im, eigvecs_im = np.linalg.eigh(np.moveaxis(Psi_w_imag, -1, 0))
    # Note the issue with function is that when the eigenvalues intersect, they might be switched around
    widths = np.sqrt(2 / eigvals_im)
    width1 = np.copy(widths[:, 0])
    width2 = np.copy(widths[:, 1])

    eigvec1_x = np.copy(eigvecs_im[:, 0, 0])
    eigvec1_y = np.copy(eigvecs_im[:, 0, 1])

    eigvec2_x = np.copy(eigvecs_im[:, 1, 0])
    eigvec2_y = np.copy(eigvecs_im[:, 1, 1])

    # Following code is to switch the eigenvalues/eigenvalues back
    gradgrad_width1 = np.gradient(
        np.gradient(width1, distance_along_line), distance_along_line
    )

    idx_switch_im = np.argmax(gradgrad_width1)

    if gradgrad_width1[idx_switch_im] > thresh_gradgrad * np.mean(gradgrad_width1):
        if (
            abs(width1[idx_switch_im] - width2[idx_switch_im])
            < np.mean(width1) / thresh_intersect
        ):
            for i in range(idx_switch_im, len(width1)):
                width1[i] = widths[i, 1]
                width2[i] = widths[i, 0]
                eigvec1_x[i] = eigvecs_im[i, 1, 0]
                eigvec1_y[i] = eigvecs_im[i, 1, 1]
                eigvec2_x[i] = eigvecs_im[i, 0, 0]
                eigvec2_y[i] = eigvecs_im[i, 0, 1]

    # principal vectors in terms of Lab Cartesian Coordinates

    pvec1 = np.zeros((numberOfDataPoints, 3))
    pvec2 = np.zeros((numberOfDataPoints, 3))

    for i in range(numberOfDataPoints):
        pvec1[i] = width1[i] * (
            eigvec1_x[i] * x_hat_Cartesian[i] + eigvec1_y[i] * y_hat_Cartesian[i]
        )
        pvec2[i] = width2[i] * (
            eigvec2_x[i] * x_hat_Cartesian[i] + eigvec2_y[i] * y_hat_Cartesian[i]
        )

    curv1 = np.zeros([numberOfDataPoints])
    curv2 = np.zeros([numberOfDataPoints])

    eigvals_re, eigvecs_re = np.linalg.eigh(np.moveaxis(Psi_w_real, -1, 0))

    curv1 = (eigvals_re[:, 0] / K_magnitude_array) * (
        np.cos(theta_output + theta_m_output)
    ) ** 2
    curv2 = (eigvals_re[:, 1] / K_magnitude_array) * (
        np.cos(theta_output + theta_m_output)
    ) ** 2
    curv1_temp = np.copy(curv1)
    curv2_temp = np.copy(curv2)

    gradgrad_curv1 = np.gradient(
        np.gradient(curv1, distance_along_line), distance_along_line
    )

    idx_switch_re = np.argmax(gradgrad_curv1)

    if gradgrad_curv1[idx_switch_re] > thresh_gradgrad * np.mean(gradgrad_curv1):
        if (
            abs(curv1[idx_switch_re] - curv2[idx_switch_re])
            < np.mean(np.abs(curv1)) / thresh_intersect
        ):
            for i in range(idx_switch_re, len(curv1)):
                curv1[i] = curv2_temp[i]
                curv2[i] = curv1_temp[i]

    # ==================================================================================#
    # Setting the Main Plot
    px = 1 / plt.rcParams["figure.dpi"]  # defining pixels
    fig = plt.figure(
        tight_layout=True, figsize=(1600 * px, 800 * px), num="3D Beam Propagation"
    )

    title = ""
    if include_title:
        title += f"\nshot = {title_shot}"
        title += f"\ntime = {title_time}ms"
        title += f"\nmode = {title_mode}"
        title += f"\nfreq = {title_freq:.1f}GHz"
        title += f"\npol = {title_pol:.1f}\u00B0"
        title += f"\ntor = {title_tor:.1f}\u00B0"
        title += f"\nwidth = {title_width:.3f}m"
        title += f"\ncurv = {title_curv:.3f}" + r"$m^{-1}$"
        title += f"\n{title_additional_notes}"

    fig.text(
        0.05, 0.78, title, ha="center"
    )  # Note the positions are for the bottom left corner of the text

    # fig refers to the entire window, each axes refers to a specific graph within the entire figure

    ax = plt.subplot2grid(
        (4, 8), (0, 0), colspan=4, rowspan=4, projection="3d"
    )  # initialise the plot

    ax.plot(
        q_X_array, q_Y_array, q_Z_array, label="Central Ray", lw=5, color="blue"
    )  # plot the trajectory of the central ray

    ax.scatter(
        [q_X_array[0]], [q_Y_array[0]], [q_Z_array[0]], marker="o", s=50, color="black"
    )  # plots starting point

    arrow_vec = g_hat_Cartesian[out_index]
    ax.quiver(
        q_X_array[out_index],
        q_Y_array[out_index],
        q_Z_array[out_index],
        arrow_vec[0],
        arrow_vec[1],
        arrow_vec[2],
        length=0.01,
        arrow_length_ratio=3,
        linewidth=5.0,
        normalize=True,
        color="blue",
    )  # add an arrow at the end of the ray trajectory to tell users how ray is travelling

    ##resizing stuff
    length_required = np.array(
        [
            np.max(q_X_array) - np.min(q_X_array),
            np.max(q_Y_array) - np.min(q_Y_array),
            np.max(q_Z_array) - np.min(q_Z_array),
        ]
    )
    max_width = np.max(length_required)

    x_window = (max_width - length_required[0]) / 2 + window_size
    y_window = (max_width - length_required[1]) / 2 + window_size
    z_window = (max_width - length_required[2]) / 2 + window_size

    # Subsequent settings are to ensure that the lengths are the same

    ax.set_xlim(np.min(q_X_array) - x_window, np.max(q_X_array) + x_window)
    ax.set_ylim(np.min(q_Y_array) - y_window, np.max(q_Y_array) + y_window)
    ax.set_zlim(np.min(q_Z_array) - z_window, np.max(q_Z_array) + z_window)

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")
    ax.legend()

    # ==================================================================================#
    # Plotting the beam surface
    # We can use the coordinates here for the interactive ellipse to be plotted

    X_beam_surface = np.zeros([numberOfDataPoints, ellipse_resolution + 1])
    Y_beam_surface = np.zeros([numberOfDataPoints, ellipse_resolution + 1])
    Z_beam_surface = np.zeros([numberOfDataPoints, ellipse_resolution + 1])

    theta = np.concatenate(
        (np.arange(0, 2 * np.pi, (2 * np.pi / ellipse_resolution)), [0.0]), axis=0
    )

    cos_theta_array = np.cos(theta)
    sin_theta_array = np.sin(theta)

    for idx1 in range(numberOfDataPoints):
        for idx2 in range(ellipse_resolution + 1):

            w1 = pvec1[idx1] * cos_theta_array[idx2]
            w2 = pvec2[idx1] * sin_theta_array[idx2]

            X_beam_surface[idx1][idx2] = w1[0] + w2[0] + q_X_array[idx1]
            Y_beam_surface[idx1][idx2] = w1[1] + w2[1] + q_Y_array[idx1]
            Z_beam_surface[idx1][idx2] = w1[2] + w2[2] + q_Z_array[idx1]

    ax.plot_surface(
        X_beam_surface,
        Y_beam_surface,
        Z_beam_surface,
        edgecolor="royalblue",
        lw=0.04,
        rstride=4,
        cstride=4,
        alpha=0.01,
    )  # plot the beam surface

    # ==================================================================================#
    # Plotting the x and y hat planes
    # We first find the magnitude to be plotted, which we want to be such that this plane, stretches exactly to the beam_surface
    # The is some simple trigo
    mag_x_plane = np.sqrt((width1 * eigvec1_x) ** 2 + (width2 * eigvec2_x) ** 2)
    mag_y_plane = np.sqrt((width1 * eigvec1_y) ** 2 + (width2 * eigvec2_y) ** 2)

    x_plane = np.zeros([numberOfDataPoints, 2, 3])
    y_plane = np.zeros([numberOfDataPoints, 2, 3])

    for idx1 in range(numberOfDataPoints):
        cray_pos = np.array(
            [q_X_array[idx1], q_Y_array[idx1], q_Z_array[idx1]]
        )  # I couldnt think of a better name, but its simply central ray position
        x_plane[idx1][0] = -mag_x_plane[idx1] * x_hat_Cartesian[idx1] + cray_pos
        x_plane[idx1][1] = mag_x_plane[idx1] * x_hat_Cartesian[idx1] + cray_pos
        y_plane[idx1][0] = -mag_y_plane[idx1] * y_hat_Cartesian[idx1] + cray_pos
        y_plane[idx1][1] = mag_y_plane[idx1] * y_hat_Cartesian[idx1] + cray_pos

    colorx = "green"
    colory = "red"

    ax.plot_surface(
        x_plane[:, :, 0],
        x_plane[:, :, 1],
        x_plane[:, :, 2],
        edgecolor=colorx,
        lw=0.04,
        rstride=4,
        cstride=4,
        color=colorx,
        alpha=0.1,
    )  # plot the x_hat plane
    ax.plot_surface(
        y_plane[:, :, 0],
        y_plane[:, :, 1],
        y_plane[:, :, 2],
        edgecolor=colory,
        lw=0.04,
        rstride=4,
        cstride=4,
        color=colory,
        alpha=0.1,
    )  # plot the y_hat plane
    # ==================================================================================#
    # Setting the required functions
    # For the initial poss, and slider updating

    def index_pos(distance):
        # print(np.abs(distance_along_line - distance))
        index_pos = np.argmin(np.abs(distance_along_line - distance))
        return index_pos

    def dot_pos(distance):
        index_pos = np.argmin(np.abs(distance_along_line - distance))
        return [q_X_array[index_pos], q_Y_array[index_pos], q_Z_array[index_pos]]

    # ==================================================================================#
    # Initial conditions

    init_distance = distance_along_line[cutoff_index]

    # Dot on the path
    (dot,) = ax.plot(
        dot_pos(init_distance)[0],
        dot_pos(init_distance)[1],
        dot_pos(init_distance)[2],
        "ro",
        markersize=10,
        color="black",
    )  # initialise the moveable point to be at the cutoff point

    (dot_cutoff,) = ax.plot(
        dot_pos(init_distance)[0],
        dot_pos(init_distance)[1],
        dot_pos(init_distance)[2],
        "ro",
        markersize=15,
        color="red",
    )  # mark out where the cutoff point is located

    pvec1_init_pos = pvec1[index_pos(init_distance)]
    pvec2_init_pos = pvec2[index_pos(init_distance)]

    # Line linking dot on the path to dot pointing along pvec1
    line_pvec1_init_pos = [
        (
            dot_pos(init_distance)[0] - pvec1_init_pos[0],
            dot_pos(init_distance)[1] - pvec1_init_pos[1],
            dot_pos(init_distance)[2] - pvec1_init_pos[2],
        ),
        (
            dot_pos(init_distance)[0] + pvec1_init_pos[0],
            dot_pos(init_distance)[1] + pvec1_init_pos[1],
            dot_pos(init_distance)[2] + pvec1_init_pos[2],
        ),
    ]
    line_pvec1 = ax.plot(*zip(*line_pvec1_init_pos))[0]
    line_pvec1.set_color("green")
    line_pvec1.set_linewidth(3.0)

    # Line linking dot on the path to dot pointing along pvec2
    line_pvec2_init_pos = [
        (
            dot_pos(init_distance)[0] - pvec2_init_pos[0],
            dot_pos(init_distance)[1] - pvec2_init_pos[1],
            dot_pos(init_distance)[2] - pvec2_init_pos[2],
        ),
        (
            dot_pos(init_distance)[0] + pvec2_init_pos[0],
            dot_pos(init_distance)[1] + pvec2_init_pos[1],
            dot_pos(init_distance)[2] + pvec2_init_pos[2],
        ),
    ]
    line_pvec2 = ax.plot(*zip(*line_pvec2_init_pos))[
        0
    ]  # i dont quite know whats the [0] doing, so Ill leave it there
    line_pvec2.set_color("red")
    line_pvec2.set_linewidth(3.0)

    # ==================================================================================#

    # ellipse to be interactive
    def totuple(array):
        try:
            return tuple(totuple(i) for i in array)
        except TypeError:
            return array

    ellipse_init_pos = np.zeros([ellipse_resolution + 1], dtype=object)
    idx_init = index_pos(init_distance)
    for idx in range(ellipse_resolution + 1):
        ellipse_init_pos[idx] = totuple(
            np.array(
                [
                    X_beam_surface[idx_init][idx],
                    Y_beam_surface[idx_init][idx],
                    Z_beam_surface[idx_init][idx],
                ]
            )
        )

    ellipse_int = ax.plot(*zip(*ellipse_init_pos))[
        0
    ]  # i dont quite know whats the [0] doing, so Ill leave it there
    ellipse_int.set_color("blue")
    ellipse_int.set_linewidth(3.0)

    # ==================================================================================#

    # Plotting the cross-section of the beam

    ax2 = plt.subplot2grid((4, 8), (0, 4), colspan=2, rowspan=2)

    def pvec1_2d_pos(idx):
        return np.array([width1[idx] * eigvec1_x[idx], width1[idx] * eigvec1_y[idx]])

    def pvec2_2d_pos(idx):
        return np.array([width2[idx] * eigvec2_x[idx], width2[idx] * eigvec2_y[idx]])

    window_2d_plot = np.max(np.array([np.max(width1), np.max(width2)]))

    ax2.set_xlim(-window_2d_plot - window_size, window_2d_plot + window_size)
    ax2.set_ylim(-window_2d_plot - window_size, window_2d_plot + window_size)

    ax2.set_xlabel("x/m")
    ax2.set_ylabel("y/m")

    pvec1_2d = pvec1_2d_pos(idx_init)
    (line_pvec1_2d,) = ax2.plot(
        [-pvec1_2d[0], pvec1_2d[0]],
        [-pvec1_2d[1], pvec1_2d[1]],
        color="green",
        linewidth=3.0,
    )
    pvec2_2d = pvec2_2d_pos(idx_init)
    (line_pvec2_2d,) = ax2.plot(
        [-pvec2_2d[0], pvec2_2d[0]],
        [-pvec2_2d[1], pvec2_2d[1]],
        color="red",
        linewidth=3.0,
    )

    xline_ax2 = ax2.axvline(x=0, color=colory, lw=0.5)
    yline_ax2 = ax2.axhline(y=0, color=colory, lw=0.5)

    ellipse_init_pos_x = np.zeros(ellipse_resolution + 1)
    ellipse_init_pos_y = np.zeros(ellipse_resolution + 1)

    for idx in range(ellipse_resolution + 1):

        ellipse_init_pos_x[idx] = (
            cos_theta_array[idx] * pvec1_2d[0] + sin_theta_array[idx] * pvec2_2d[0]
        )
        ellipse_init_pos_y[idx] = (
            cos_theta_array[idx] * pvec1_2d[1] + sin_theta_array[idx] * pvec2_2d[1]
        )

    (ellipse_int_2d,) = ax2.plot(
        ellipse_init_pos_x, ellipse_init_pos_y, color="blue", linewidth=3.0
    )

    # ==================================================================================#
    # Plot for the widths

    # 2D width plot
    ax3 = plt.subplot2grid((4, 8), (2, 4), colspan=2, rowspan=2)

    ax3.set_xlabel("distance/m")
    ax3.set_ylabel("width/m")
    ax3.set_xlim(
        min(distance_along_line) - window_size, max(distance_along_line) + window_size
    )
    if include_width1:
        ax3.plot(distance_along_line, width1, lw=2, color="green", label="width1")

    if include_width2:
        ax3.plot(distance_along_line, width2, lw=2, color="red", label="width2")

    lambda_array = (2 * np.pi) / K_magnitude_array

    ax3.plot(
        distance_along_line,
        lambda_array,
        lw=2,
        dashes=[4, 2],
        color="blue",
        label="wavelength",
    )  # plot how the wavelength changes

    line_ax3_cutoff = ax3.axvline(x=init_distance, color="orange", lw=1.5)
    line_ax3 = ax3.axvline(x=init_distance, color="black", lw=1.5)
    ax3.legend(
        fontsize=10,
        loc="upper left",
        # bbox_to_anchor=(0.0, 0.3)
    )

    # ==================================================================================#
    # Plot for the curvature

    ax4 = plt.subplot2grid((4, 8), (2, 6), colspan=2, rowspan=2)

    ax4.set_xlabel("distance/m")
    ax4.set_ylabel("curvature/(1/m)")
    ax4.set_xlim(
        min(distance_along_line) - window_size, max(distance_along_line) + window_size
    )

    ax4.plot(distance_along_line, curv1, lw=2, color="blue", label="curv1")
    ax4.plot(distance_along_line, curv2, lw=2, color="maroon", label="curv2")

    line_ax4_cutoff = ax4.axvline(x=init_distance, color="orange", lw=1.5)
    line_ax4 = ax4.axvline(x=init_distance, color="black", lw=1.5)
    ax4.legend(fontsize=10, loc="upper left", bbox_to_anchor=(0.0, 1.0))

    # ==================================================================================#
    # Plot for the localisation

    ax5 = plt.subplot2grid((4, 8), (0, 6), colspan=2, rowspan=2)
    line_ax5 = ax5.axvline(x=init_distance, color="black", lw=1.5)

    ax5.set_xlabel("distance/m")
    ax5.set_ylabel("localisation")
    # ax5.set_xlim(min(distance_along_line)-window_size, max(distance_along_line)+window_size)

    ##labelling localisations and multiplying localisation terms together
    resultant_loc = 1
    name = "loc"
    if include_p:
        name += "_p"
        resultant_loc *= loc_p
    if include_s:
        name += "_s"
        resultant_loc *= loc_s
    if include_m:
        name += "_m"
        resultant_loc *= loc_m
    if include_b:
        name += "_b"
        resultant_loc *= loc_b
    if include_r:
        name += "_r"
        resultant_loc = resultant_loc * loc_r

    ax5.plot(distance_along_line, resultant_loc, lw=2, color="red", label=name)

    line_ax5_cutoff = ax5.axvline(x=init_distance, color="orange", lw=1.5)
    line_ax5 = ax5.axvline(x=init_distance, color="black", lw=1.5)
    ax5.legend(
        fontsize=10,
        loc="upper left",
        # bbox_to_anchor=(0.0, 0.3)
    )

    # ==================================================================================#
    # Setting dimensions of slider
    ax_slider = plt.axes(
        [0.08, 0.01, 0.35, 0.03]
    )  # 1st 2 arguments are pos, next 2 are height and width in some order, quantity are ratios wrt to figure proportion

    distance_slider = Slider(
        ax_slider,
        label="Distance[m]",
        valmin=0.0,
        valmax=distance_along_line[out_index],
        valinit=init_distance,
    )

    # update function for the slider
    def update(val):
        pos = dot_pos(val)
        idx = index_pos(val)
        pvec1a = pos - pvec1[idx]
        pvec1b = pos + pvec1[idx]
        pvec2a = pos - pvec2[idx]
        pvec2b = pos + pvec2[idx]

        dot.set_xdata(pos[0])
        dot.set_ydata(pos[1])
        dot.set_3d_properties(pos[2])

        set_data_pvec1_array = np.array(
            [[pvec1a[0], pvec1b[0]], [pvec1a[1], pvec1b[1]]]
        )
        line_pvec1.set_data(set_data_pvec1_array)
        pvec1_z = (pvec1a[2], pvec1b[2])
        line_pvec1.set_3d_properties(pvec1_z)

        set_data_pvec2_array = np.array(
            [[pvec2a[0], pvec2b[0]], [pvec2a[1], pvec2b[1]]]
        )
        line_pvec2.set_data(set_data_pvec2_array)
        pvec2_z = (pvec2a[2], pvec2b[2])
        line_pvec2.set_3d_properties(pvec2_z)

        # Here is the updating for the ellipse
        set_data_ellipse_array = np.zeros([2, ellipse_resolution + 1])
        set_data_ellipse_array[0] = X_beam_surface[idx]
        set_data_ellipse_array[1] = Y_beam_surface[idx]
        ellipse_int.set_data(set_data_ellipse_array)
        ellipse_z = totuple(Z_beam_surface[idx])
        ellipse_int.set_3d_properties(ellipse_z)

        # Updating for 2Dline
        line_pvec1_2d.set_xdata([-pvec1_2d_pos(idx)[0], pvec1_2d_pos(idx)[0]])
        line_pvec1_2d.set_ydata([-pvec1_2d_pos(idx)[1], pvec1_2d_pos(idx)[1]])

        line_pvec2_2d.set_xdata([-pvec2_2d_pos(idx)[0], pvec2_2d_pos(idx)[0]])
        line_pvec2_2d.set_ydata([-pvec2_2d_pos(idx)[1], pvec2_2d_pos(idx)[1]])

        ellipse_2d_x = np.zeros(ellipse_resolution + 1)
        ellipse_2d_y = np.zeros(ellipse_resolution + 1)

        for i in range(ellipse_resolution + 1):

            ellipse_2d_x[i] = (
                cos_theta_array[i] * pvec1_2d_pos(idx)[0]
                + sin_theta_array[i] * pvec2_2d_pos(idx)[0]
            )
            ellipse_2d_y[i] = (
                cos_theta_array[i] * pvec1_2d_pos(idx)[1]
                + sin_theta_array[i] * pvec2_2d_pos(idx)[1]
            )

        ellipse_int_2d.set_xdata(ellipse_2d_x)
        ellipse_int_2d.set_ydata(ellipse_2d_y)

        line_ax3.set_xdata(val)
        line_ax4.set_xdata(val)
        line_ax5.set_xdata(val)

    distance_slider.on_changed(update)
    # keep the slider updated

    return ax, distance_slider
    # must keep the slider globally in order for it to respond. Store it in any variable should be sufficient


def plot_psi(dt: DataTree, filename: Optional[PathLike] = None):
    """
    Plots the following:
    1. Poloidal flux vs tau
    2. Real and imaginary part of psi_xx vs tau
    3. Real and imaginary part of psi_xy vs tau
    4. Real and imaginary part of psi_yy vs tau
    """
    plt.figure()
    plt.rcParams.update({"axes.titlesize": "small"})

    plt.subplot(2, 2, 1)
    plt.plot(dt.analysis.tau, dt.analysis.poloidal_flux)  ##plot poloidal flux
    plt.title("Poloidal flux", pad=-5)

    plt.subplot(2, 2, 2)
    plt.plot(dt.analysis.tau, dt.analysis.Psi_xx.real, label="re")  ##real part psi_xx
    plt.plot(dt.analysis.tau, dt.analysis.Psi_xx.imag, label="im")  ##im part psi_xx
    plt.title("psi_xx", pad=-5)

    plt.subplot(2, 2, 3)
    plt.plot(dt.analysis.tau, dt.analysis.Psi_xy.real, label="re")  ##real part psi_xy
    plt.plot(dt.analysis.tau, dt.analysis.Psi_xy.imag, label="im")  ##im part psi_xy
    plt.title("psi_xy", pad=-5)

    plt.subplot(2, 2, 4)
    plt.plot(dt.analysis.tau, dt.analysis.Psi_yy.real, label="re")  ##re part psi_yy
    plt.plot(dt.analysis.tau, dt.analysis.Psi_yy.imag, label="im")  ##im part psi_yy
    plt.title("psi_yy", pad=-5)

    plt.legend(loc="lower right")

    if filename:
        plt.savefig(f"{filename}.png", dpi=1200)  # save figure
