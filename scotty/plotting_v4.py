import numpy as np
import plotly.graph_objects as pgo
import plotly.io as pio
from scotty.geometry_v4 import InterpolatedField_Cartesian, MagneticField_Cartesian, MagneticField_Cylindrical
from scotty.typing import FloatArray
from typing import Union, Tuple, Optional, cast

# TODO implement original scotty plotting functions



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

    Z = field.Z_coord if zlims is None else np.linspace(*zlims) # type: ignore

    if isinstance(field, (MagneticField_Cartesian, InterpolatedField_Cartesian)):
        X = field.X_coord if xlims is None else np.linspace(*xlims) # type: ignore
        Y = field.Y_coord if ylims is None else np.linspace(*ylims) # type: ignore
        XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing="ij")
        flux_coord = np.sqrt(field.polflux_cartesian(XX, YY, ZZ)) if use_rho else field.polflux_cartesian(XX, YY, ZZ)
        flux_coord = cast(FloatArray, flux_coord)

    else: # elif isinstance(field, MagneticField_Cylindrical, InterpolatedField_Cylindrical):
        X = field.R_coord if xlims is None else np.linspace(*xlims) # type: ignore
        Y = np.linspace(-0.5, 0.5, 41) if ylims is None else np.linspace(*ylims) # type: ignore
        XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing="ij")
        flux_coord = np.sqrt(field.poloidal_flux(np.sqrt(XX**2 + YY**2), ZZ)) if use_rho else field.poloidal_flux(np.sqrt(XX**2 + YY**2), ZZ)
        flux_coord = cast(FloatArray, flux_coord)
    
    # masking to remove NaNs and outside LCFS, to see if we can speed up plotting time
    # flux_threshold = 1.0 + 0.1 # leave this here just in case we want to implement this in the future
    mask = np.isfinite(flux_coord) # & (flux_coord <= flux_threshold) # masking like this prevents plotly from plotting, see if can fix
    XX = XX[mask]
    YY = YY[mask]
    ZZ = ZZ[mask]
    flux_coord = flux_coord[mask]

    fig = make_fig_if_None(existing_fig)

    fig.add_trace(pgo.Isosurface(
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
        name=f"Flux surfaces, {r'$\rho_n' if use_rho else r'$\psi_n$'}",
        colorbar={
            "title": r"$\rho_n$" if use_rho else r"$\psi_n$",
            "thickness": 10, "len": 0.7,
        }
    ))

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
        height=700
    )

    if show_graph:
        if show_in_browser: pio.renderers.default = "browser"
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
    q_XX = q_X[:, None] # shape (N,1)
    q_YY = q_Y[:, None] # shape (N,1)
    q_ZZ = q_Z[:, None] # shape (N,1)
    w_NN = widths_N[:, None] # shape (N,1)
    w_BB = widths_B[:, None] # shape (N,1)

    n_T = n_hat.T # shape (3,N)
    b_T = b_hat.T # shape (3,N)

    theta = np.linspace(0, 2*np.pi, theta_resolution)
    sin_theta = np.sin(theta)[None, :] # shape (1,T)
    cos_theta = np.cos(theta)[None, :] # shape (1,T)

    X_beam_surf = (q_XX + (w_NN * cos_theta * n_T[0][:, None]) + (w_BB * sin_theta * b_T[0][:, None])).ravel()
    Y_beam_surf = (q_YY + (w_NN * cos_theta * n_T[1][:, None]) + (w_BB * sin_theta * b_T[1][:, None])).ravel()
    Z_beam_surf = (q_ZZ + (w_NN * cos_theta * n_T[2][:, None]) + (w_BB * sin_theta * b_T[2][:, None])).ravel()

    # Get triangular mesh indices for the corresponding points
    # of the beam silhouette, where the vertices are connected like
    # triangle set 1: (i, j) -> (i, j+1) -> (i+1, j+1)
    # triangle set 2: (i, j) -> (i+1, j+1) -> (i+1, j)
    # and stack them all together
    ii = np.arange(len(q_X))
    jj = np.arange(theta_resolution)
    I, J = np.meshgrid(ii[:-1], jj[:-1], indexing="ij")
    I, J = I.ravel(), J.ravel()

    def tri_map(i, j): return i*theta_resolution + j
    t_i = np.concatenate([tri_map(I, J),     tri_map(I, J)])
    t_j = np.concatenate([tri_map(I, J+1),   tri_map(I+1, J+1)])
    t_k = np.concatenate([tri_map(I+1, J+1), tri_map(I+1, J)])

    # Some plotly stuff
    fig = make_fig_if_None(existing_fig)

    # Plot central ray
    fig.add_trace(pgo.Scatter3d(
        x=q_X,
        y=q_Y,
        z=q_Z,
        mode="lines",
        line=dict(color="black", width=6),
        name="Central ray (plasma)"
    ))

    # Plot beam surface
    fig.add_trace(pgo.Mesh3d(
        x=X_beam_surf,
        y=Y_beam_surf,
        z=Z_beam_surf,
        i=t_i,
        j=t_j,
        k=t_k,
        color="black",
        opacity=0.25,
        name="1/e Beam (plasma)"
    ))

    if show_graph:
        if show_in_browser: pio.renderers.default = "browser"
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