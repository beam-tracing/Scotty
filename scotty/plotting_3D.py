import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pathlib
from scotty.geometry_3D import InterpolatedField_3D_Cartesian
from scotty.typing import PathLike, FloatArray
from typing import Optional, Union
import xarray as xr

# TO REMOVE: all trailing commas



def maybe_make_axis(ax: Optional[plt.Axes], *args, **kwargs) -> plt.Axes:
    if ax is None: _, ax = plt.subplots(*args, **kwargs)
    return ax



def get_metadata(inputs: xr.Dataset) -> dict:
    
    # The items are from an xarray
    # So they are either floats/ints or strings
    # Parse them as such
    def format_metadata(item):
        try: newitem = f"{np.round(np.array(item), 3)}"
        except: newitem = str(item)
        else: newitem = f"{np.round(np.array(item), 3)}"
        return newitem
    
    # TO REMOVE -- probably can get units directly from dt.attrs or something?
    return {
        "Poloidal Launch Angle": format_metadata(inputs.poloidal_launch_angle_Torbeam.data) + r"$^\circ$",
        "Toroidal Launch Angle": format_metadata(inputs.toroidal_launch_angle_Torbeam.data) + r"$^\circ$",
        "Launch Position [X, Y, Z]": format_metadata(inputs.launch_position_cartesian.data),
        "Launch Frequency": format_metadata(inputs.launch_freq_GHz.data) + " GHz",
        "Launch Beam Width": format_metadata(inputs.launch_beam_width.data) + "m",
        "Launch Beam Curvature": format_metadata(inputs.launch_beam_curvature.data) + r"m$^{-1}$",
        "Mode Flag": format_metadata(inputs.mode_flag.data),
    }



# TO REMOVE -- idk how to streamline everything into plotter() yet
def add_metadata_plot(metadata: dict,
                      old_fig: plt.Figure,
                      old_axs: Union[plt.Axes, list[plt.Axes]] = None):
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    
    metadata_fontsize = float(metadata.pop("fontsize", 16))
    for metadata_name, metadata_vals in metadata.items(): ax.plot(-1, -1, label=f"{metadata_name} = {metadata_vals}", color="black", marker="o", linestyle="")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc="center left", fontsize=metadata_fontsize)
    ax.axis("off")

    new_fig = old_fig.add_subplot(122)

    return new_fig



def plotter(all_X_data: Union[dict, list[dict]],
            all_Y_data: Union[dict, list[dict], list[list[dict]]],
            metadata: dict,
            filename: Union[str, pathlib.Path],
            # old_axs: Optional[Union[plt.Axes, list[plt.Axes]]] = None,
            figure_title: str = "",
            figure_title_fontsize: float = 20,
            figure_size = 6,
            metadata_flag = True,
            metadata_fontsize: float = 16):
    
    # Making sure the file name and directory are valid
    filename = pathlib.Path(filename)

    # The variables must contain data
    if len(all_X_data) == 0: raise ValueError("Empty X-dataset passed!")
    if len(all_Y_data) == 0: raise ValueError("Empty Y-dataset passed!")
    
    # Ensuring the data is in a standard format
    # The end result is that:
    #   `all_X_data` is now always of the type list[dict], and
    #   `all_Y_data` is now always of the type list[list[dict]]
    if isinstance(all_X_data, dict): all_X_data = [all_X_data]
    if isinstance(all_Y_data, dict): all_Y_data = [[all_Y_data]]
    elif isinstance(all_Y_data, list) and isinstance(all_Y_data[0], dict): all_Y_data = [all_Y_data]

    # When `all_X_data` contains only 1 dict, then it doesn't matter how many `all_Y_data` has
    #   This corresponds to the case of 1 subplot with either 1 graph or N graphs
    # But when `all_X_data` contains N dicts, then `all_Y_data` must either also be a list containing N dicts, or
    #   a list N lists each containing however many dicts
    #   This corresponds to the case of, respectively, N subplots with 1 graph or N subplots with however many graphs each
    # In both cases, this must mean that either `all_X_data` has only 1 dict, or `all_X_data` and `all_Y_data` must have
    #   the same length
    len_all_X_data = len(all_X_data)
    len_all_Y_data = len(all_Y_data)
    if len_all_X_data != 1 and len_all_X_data != len_all_Y_data: raise ValueError(f"{len_all_X_data} X-datasets passed for {len_all_Y_data} subplots!")

    # If `all_Y_data` contains N lists, then the user wants N subplots
    # Otherwise, `all_Y_data` contains just 1 list
    num_subplots = len_all_Y_data
    
    # Handle case where all subplots use the same X data
    use_same_X_data = len_all_X_data == 1

    # Creating the subplots
    fig, axs = plt.subplots(1, num_subplots+metadata_flag, figsize=(figure_size*(num_subplots+metadata_flag), figure_size+1))

    # Looping over the subplots
    for i in range(num_subplots):
        # X data for each subplot
        # The variable name must always either be the first key or the label (if specified)
        X_data_key = list(all_X_data[0 if use_same_X_data else i].keys())[0]
        X_data_vals = all_X_data[0][X_data_key] if use_same_X_data else all_X_data[i][X_data_key]

        # The very first dictionary are the details of the subplot
        # We pop it because the remaining dictionaries are passed
        # into the plotting routines as kwargs
        subplot_information = all_Y_data[i].pop(0)

        num_subgraphs = len(all_Y_data[i])
        for j in range(num_subgraphs):
            # Looping over Y data
            # The variable name must always be the first key
            Y_data_key = list(all_Y_data[i][j].keys())[0]
            Y_data_title = Y_data_key # if "label" not in all_Y_data[i][j].keys() else all_Y_data[i][j].pop("label") # TO REMOVE???
            # Pop it off the list, then pass the remainder to plotting kwargs
            Y_data_vals = all_Y_data[i][j].pop(Y_data_key)
            axs[i].plot(X_data_vals, Y_data_vals, label=Y_data_title, **all_Y_data[i][j])
        
        subplot_title = subplot_information.pop("subplot_title", False)
        subplot_title_fontsize = subplot_information.pop("subplot_title_fontsize", 16)
        axs[i].set_title(subplot_title if subplot_title else None,
                         fontsize = subplot_title_fontsize)

        subplot_X_title = X_data_key
        subplot_XY_title_fontsize = subplot_information.pop("subplot_XY_title_fontsize", 12)
        axs[i].set_xlabel(subplot_X_title if subplot_X_title else None,
                          fontsize = subplot_XY_title_fontsize)

        subplot_Y_title = subplot_information.pop("subplot_Y_title", False)
        axs[i].set_ylabel(Y_data_title if subplot_Y_title else None,
                          fontsize = subplot_XY_title_fontsize)
        
        subplot_aspect_ratio = subplot_information.pop("subplot_square_size", 1)
        axs[i].set_box_aspect(subplot_aspect_ratio)
        
        subplot_legend = subplot_information.pop("subplot_legend", True)
        subplot_legend_fontsize = subplot_information.pop("subplot_legend_fontsize", 12)
        axs[i].legend(fontsize = subplot_legend_fontsize) if subplot_legend else None

        subplot_grid = subplot_information.pop("subplot_grid", False)
        axs[i].grid(subplot_grid)
    
    # For last subplot, put all details of the simulation
    if metadata_flag:
        metadata_fontsize = float(metadata.pop("fontsize", metadata_fontsize))
        for metadata_name, metadata_vals in metadata.items(): axs[-1].plot(-1, -1, label=f"{metadata_name} = {metadata_vals}", color="black", marker="o", linestyle="")
        axs[-1].set_xlim(0,1)
        axs[-1].set_ylim(0,1)
        axs[-1].legend(loc="center left", fontsize=metadata_fontsize)
        axs[-1].axis("off")

    # Modifying the entire figure
    if figure_title: plt.suptitle(figure_title,
                                  fontsize = figure_title_fontsize,
                                  fontweight = "bold")

    plt.tight_layout()
    plt.draw()
    if filename: plt.savefig(f"{filename}")
    plt.close()

    return fig, axs



def plot_2d_cross_section(
    field: InterpolatedField_3D_Cartesian,
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    X_data: Union[float, list, FloatArray] = None,
    Y_data: Union[float, list, FloatArray] = 0.0,
    Z_data: Union[float, list, FloatArray] = None,
    zoom = "standard"): # TO REMOVE -- not implemeted yet

    # If the user does not specify bounding box, take it from the field
    # If all of them are None, then just slice in Y
    all_data = {"X-coordinate": X_data if X_data is not None else field.X_coord,
                "Y-coordinate": Y_data if Y_data is not None else field.Y_coord,
                "Z-coordinate": Z_data if Z_data is not None else np.linspace(-1, 1, 201)} # field.Z_coord}
    
    # Separating the array data into its own dictionary, and the float
    # data as a standalone float
    float_data = {}
    array_data = {}
    for name, data in all_data.items():
        # The coordinates must be in list or array form, with a dimension of 1, and with more than 1 item
        # or it must be a float, otherwise raise an error
        if   isinstance(data, float): float_data[name] = data
        elif isinstance(data, (list, np.ndarray)) and len(np.array(data).shape) == 1 and len(data) != 1: array_data[name] = data
        else: ValueError(f"plot_2d_cross_section: Must specify only 1 coordinate and 2 arrays for a 2d cross-section!")
    
    # `zoom` must be valid
    _valid_zoom = ["in", "standard", "out"]
    if zoom not in _valid_zoom:
        print(f"plot_2d_cross_section: {zoom} is not a valid plot zoom option. Setting `zoom` = 'standard'")
        zoom = "standard"
    
    # Creating the interpolation grid
    horizontal_data_name, vertical_data_name = array_data.keys()
    horizontal_data,      vertical_data =      array_data.values()
    hh, vv = np.meshgrid(horizontal_data, vertical_data, indexing="ij")
    if   "X-coordinate" in float_data.keys(): polflux_data = field.polflux(list(float_data.values()), hh, vv)
    elif "Y-coordinate" in float_data.keys(): polflux_data = field.polflux(hh, list(float_data.values()), vv)
    elif "Z-coordinate" in float_data.keys(): polflux_data = field.polflux(hh, vv, list(float_data.values()))
    else: raise ValueError("plot_2d_cross_section: Not sure how to slice the 2d data!")

    # Creating the plot
    fig, axs = plt.subplots(1, 2, figsize=(6*2, 6+1))
    axs[0].set_aspect("equal", adjustable='box')

    # Plotting the contours
    cax = axs[0].contour(horizontal_data, vertical_data,
                         polflux_data.T, # matrix indexed as "ij" instead of "xy"
                         levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
                         cmap = "plasma_r")
    
    # Adding contour labels
    axs[0].clabel(cax, inline=True, fontsize=8, inline_spacing=10, fmt='%1.1f')

    # Adding axis labels
    axs[0].set_xlabel(f"{horizontal_data_name}")
    axs[0].set_ylabel(f"{vertical_data_name}")

    # Adding metadata subplot
    metadata = get_metadata(inputs)

    if metadata:
        metadata_fontsize = float(metadata.pop("fontsize", 16))
        for metadata_name, metadata_vals in metadata.items(): axs[-1].plot(-1, -1, label=f"{metadata_name} = {metadata_vals}", color="black", marker="o", linestyle="")
        axs[-1].set_xlim(0,1)
        axs[-1].set_ylim(0,1)
        axs[-1].legend(loc="center left", fontsize=metadata_fontsize)
        axs[-1].axis("off")

    # TO REMOVE -- this doesnt work :(
    # newfig = add_metadata_plot(metadata, fig)

    plt.tight_layout()
    plt.draw()
    if filename: plt.savefig(f"{filename}")
    plt.close()
    
    return fig, axs



# def plot_curvatures(): # TO REMOVE need to implement this properly
#     return



def plot_delta_theta_m(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None):
    
    metadata = get_metadata(inputs)

    data_Xaxis = {"Arc length relative to cut-off (m)": np.array(analysis.arc_length_relative_to_cutoff)}

    data_Yaxis = [ [ {"subplot_title": f"",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"delta_theta_m": np.array(analysis.delta_theta_m)},
                   ],
                 ]

    fig, axs = plotter(data_Xaxis, data_Yaxis, metadata, filename,
                       figure_title = r"$\Delta\theta_m$",
                       # figure_title_fontsize = 
                       # figure_size = 
                       # metadata_flag = 
                       # metadata_fontsize = 
                       )

    return fig, axs



def plot_dispersion_relation(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None):
    """
    Plots Cardano's and np.linalg's solutions to the actual dispersion relation
    Useful to check whether the solution which = 0 along the path changes
    """

    metadata = get_metadata(inputs)

    data_Xaxis = {"Arc length relative to cut-off (m)": np.array(analysis.arc_length_relative_to_cutoff)}

    data_Yaxis = [ [ {"subplot_title": f"",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": True,
                      "subplot_grid": True,
                      },
                        {"H_eigval 1 (X?)":  np.abs(np.array(analysis.H_eigvals.sel(col="X"))), "color": "pink",       "linestyle": "-", "linewidth": 4},
                        {"H eigval 2 (O?)":  np.abs(np.array(analysis.H_eigvals.sel(col="Y"))), "color": "lightgreen", "linestyle": "-", "linewidth": 4},
                        {"H eigval 3":       np.abs(np.array(analysis.H_eigvals.sel(col="Z"))), "color": "gray",       "linestyle": "-", "linewidth": 4},
                        {"H 1 Cardano":      np.abs(np.array(analysis.H_1_Cardano)),            "color": "k",          "linestyle": ":", "linewidth": 4},
                        {"H 2 Cardano (X?)": np.abs(np.array(analysis.H_2_Cardano)),            "color": "r",          "linestyle": ":", "linewidth": 4},
                        {"H 3 Cardano (O?)": np.abs(np.array(analysis.H_3_Cardano)),            "color": "g",          "linestyle": ":", "linewidth": 4},
                   ],
                 ]
    
    fig, axs = plotter(data_Xaxis, data_Yaxis, metadata, filename,
                       figure_title = r"Cardano Dispersion Relation",
                       # figure_title_fontsize = 
                       # figure_size = 
                       # metadata_flag = 
                       # metadata_fontsize = 
                       )

    return fig, axs



def plot_localisations(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None):
    
    metadata = get_metadata(inputs)
    
    data_Xaxis = {"Arc length relative to cut-off (m)": np.array(analysis.arc_length_relative_to_cutoff)}

    data_Yaxis = [ [ {"subplot_title": f"Polarisation Piece",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {" ": np.array(analysis.loc_p)},
                   ],
                   [ {"subplot_title": f"Ray Piece",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {" ": np.array(analysis.loc_r)},
                   ],
                   [ {"subplot_title": f"Beam Piece",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {" ": np.array(analysis.loc_b)},
                   ],
                   [ {"subplot_title": f"Mismatch Piece",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {" ": np.array(analysis.loc_m)},
                   ],
                   [ {"subplot_title": f"Spectrum Piece",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": True,
                      "subplot_grid": True,
                      },
                        {"k^(-10/3)": np.array(analysis.loc_s_10_3), "color": "blue"},
                        {"k^(-13/3)": np.array(analysis.loc_s_13_3), "color": "red"},
                   ],
                   [ {"subplot_title": f"All Pieces",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": True,
                      "subplot_grid": True,
                      },
                        {"k^(-10/3)": np.array(analysis.loc_all_10_3), "color": "blue"},
                        {"k^(-13/3)": np.array(analysis.loc_all_13_3), "color": "red"},
                   ], 
                 ]
    
    fig, axs = plotter(data_Xaxis, data_Yaxis, metadata, filename,
                       figure_title = r"Localisations (arb. units)",
                       # figure_title_fontsize = 
                       # figure_size = 
                       # metadata_flag = 
                       # metadata_fontsize = 
                       )
    
    return fig, axs



def plot_theta_m(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None):
    
    metadata = get_metadata(inputs)
    
    data_Xaxis = {"Arc length relative to cut-off (m)": np.array(analysis.arc_length_relative_to_cutoff)}

    data_Yaxis = [ [ {"subplot_title": f"",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"theta_m": np.array(analysis.theta_m)},
                   ],
                 ]

    fig, axs = plotter(data_Xaxis, data_Yaxis, metadata, filename,
                       figure_title = r"Mismatch Angle, $\theta_m$",
                       # figure_title_fontsize = 
                       # figure_size = 
                       # metadata_flag = 
                       # metadata_fontsize = 
                       )

    return fig, axs



def plot_trajectory(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None):

    # TO REMOVE -- how to change???
    
    data_misc = {
        "poloidal_launch_angle_Torbeam": np.round(np.array(inputs.poloidal_launch_angle_Torbeam), 3),
        "toroidal_launch_angle_Torbeam": np.round(np.array(inputs.toroidal_launch_angle_Torbeam), 3),
    }

    data_Xaxis = {
        "arc_length_relative_to_cutoff": np.array(analysis.arc_length_relative_to_cutoff),
    }
    
    data_Yaxis = {
        "q_X": np.array(solver_output.q_X),
        "q_Y": np.array(solver_output.q_Y),
        "q_Z": np.array(solver_output.q_Z),
    }

    # Plotting ray trajectory in 3d space
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter(data_Yaxis["q_X"][0], data_Yaxis["q_Y"][0], data_Yaxis["q_Z"][0], color='black', label="Start Point in Plasma")
    ax.plot(data_Yaxis["q_X"], data_Yaxis["q_Y"], data_Yaxis["q_Z"], linestyle='-', linewidth=2, zorder=1, label="Trajectory")
    # ax.set_ylim(-0.05, 0.05)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(f'Ray Trajectory, pol {data_misc["poloidal_launch_angle_Torbeam"]}, tor {data_misc["toroidal_launch_angle_Torbeam"]}')
    ax.legend()
    plt.draw()
    if filename: plt.savefig(f"{filename}")
    plt.close()



def plot_trajectories_individual(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None):
    
    metadata = get_metadata(inputs)
    
    data_Xaxis = {"Arc length relative to cut-off (m)": np.array(analysis.arc_length_relative_to_cutoff)}

    data_Yaxis = [ [ {"subplot_title": f"Ray Trajectory, X-coordinate",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"q_X (m)": np.array(solver_output.q_X)},
                   ],
                   [ {"subplot_title": f"Ray Trajectory, Y-coordinate",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"q_Y (m)": np.array(solver_output.q_Y)},
                   ],
                   [ {"subplot_title": f"Ray Trajectory, Z-coordinate",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"q_Z (m)": np.array(solver_output.q_Z)},
                   ],
                 ]
    
    fig, axs = plotter(data_Xaxis, data_Yaxis, metadata, filename,
                       figure_title = r"Individual Ray Trajectories",
                       # figure_title_fontsize = 
                       # figure_size = 
                       # metadata_flag = 
                       # metadata_fontsize = 
                       )

    return fig, axs



def plot_trajectories_poloidal(
    field: InterpolatedField_3D_Cartesian,
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    X_data: Union[float, list, FloatArray] = None,
    Y_data: Union[float, list, FloatArray] = None,
    Z_data: Union[float, list, FloatArray] = None,
    zoom = "standard"): # TO REMOVE -- not implemeted yet

    # If the user does not specify bounding box, take it from the field
    # If all of them are None, then just slice in Y, and select the coordinate
    # matching the cutoff location
    cutoff_index = int(analysis["cutoff_index"])
    all_data = {"X-coordinate": X_data if X_data is not None else field.X_coord,
                "Y-coordinate": Y_data if Y_data is not None else round(float(solver_output["q_Y"][cutoff_index]), 2),
                "Z-coordinate": Z_data if Z_data is not None else field.Z_coord}
    
    # Separating the array data into its own dictionary, and the float
    # data as a standalone float
    float_data = {}
    array_data = {}
    for name, data in all_data.items():
        # The coordinates must be in list or array form, with a dimension of 1, and with more than 1 item
        # or it must be a float, otherwise raise an error
        if   isinstance(data, float): float_data[name] = data
        elif isinstance(data, (list, np.ndarray)) and len(np.array(data).shape) == 1 and len(data) != 1: array_data[name] = data
        else: ValueError(f"plot_2d_cross_section: Must specify only 1 coordinate and 2 arrays for a 2d cross-section!")
    
    # `zoom` must be valid
    _valid_zoom = ["in", "standard", "out"]
    if zoom not in _valid_zoom:
        print(f"plot_2d_cross_section: {zoom} is not a valid plot zoom option. Setting `zoom` = 'standard'")
        zoom = "standard"
    
    # Creating the interpolation grid
    horizontal_data_name, vertical_data_name = array_data.keys()
    horizontal_data,      vertical_data =      array_data.values()
    hh, vv = np.meshgrid(horizontal_data, vertical_data, indexing="ij")
    if   "X-coordinate" in float_data.keys(): polflux_data = field.polflux(list(float_data.values()), hh, vv)
    elif "Y-coordinate" in float_data.keys(): polflux_data = field.polflux(hh, list(float_data.values()), vv)
    elif "Z-coordinate" in float_data.keys(): polflux_data = field.polflux(hh, vv, list(float_data.values()))
    else: raise ValueError("plot_2d_cross_section: Not sure how to slice the 2d data!")

    # Creating the plot
    fig, axs = plt.subplots(1, 2, figsize=(6*2, 6+1))
    axs[0].set_aspect("equal", adjustable='box')

    # Plotting the contours
    cax = axs[0].contour(horizontal_data, vertical_data,
                         polflux_data.T, # matrix indexed as "ij" instead of "xy"
                         levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
                         cmap = "plasma_r")
    
    # Adding contour labels
    axs[0].clabel(cax, inline=True, fontsize=8, inline_spacing=10, fmt='%1.1f')

    # Adding axis labels
    axs[0].set_xlabel(f"{horizontal_data_name}")
    axs[0].set_ylabel(f"{vertical_data_name}")

    # Adding metadata subplot
    metadata = get_metadata(inputs)

    if metadata:
        metadata_fontsize = float(metadata.pop("fontsize", 16))
        for metadata_name, metadata_vals in metadata.items(): axs[-1].plot(-1, -1, label=f"{metadata_name} = {metadata_vals}", color="black", marker="o", linestyle="")
        axs[-1].set_xlim(0,1)
        axs[-1].set_ylim(0,1)
        axs[-1].legend(loc="center left", fontsize=metadata_fontsize)
        axs[-1].axis("off")

    # TO REMOVE -- old code, doesn't work
    # field: InterpolatedField_3D_Cartesian,
    # inputs: xr.Dataset,
    # solver_output: xr.Dataset,
    # analysis: xr.Dataset,
    # filename: Optional[PathLike] = None,
    # ax: Optional[plt.Axes] = None):

    # metadata = get_metadata(inputs)
    
    # data_Xaxis = {"q_X (m)": np.array(solver_output.q_X)}

    # data_Yaxis = [ [ {"subplot_title": f"Ray Trajectory, Poloidal Plane",
    #                   "subplot_title_fontsize": 16,
    #                   "subplot_legend": False,
    #                   "subplot_grid": True,
    #                   },
    #                     {"q_Z (m)": np.array(solver_output.q_Z)},
    #                ],
    #              ]
    
    # fig, axs = plot_2d_cross_section(field,
    #                                  inputs,
    #                                  solver_output,
    #                                  analysis)
    
    axs[0].plot(np.array(solver_output.q_X), np.array(solver_output.q_Z), color="black", linestyle="-", linewidth=2)

    launch_position_cartesian  = np.array(inputs.launch_position_cartesian)
    initial_position_cartesian = np.array(inputs.initial_position_cartesian)
    X_coords = [launch_position_cartesian[0], initial_position_cartesian[0]]
    Z_coords = [launch_position_cartesian[2], initial_position_cartesian[2]]
    axs[0].plot(X_coords, Z_coords, color="black", linestyle="--", linewidth=2)
    plt.tight_layout()
    plt.draw()
    if filename: plt.savefig(f"{filename}")
    plt.close()

    return fig, axs



# def plot_trajectories_toroidal(
#     inputs: xr.Dataset,
#     solver_output: xr.Dataset,
#     analysis: xr.Dataset,
#     filename: Optional[PathLike] = None,
#     ax: Optional[plt.Axes] = None):

#     metadata = get_metadata(inputs)
    
#     data_Xaxis = {"Arc length relative to cut-off (m)": np.array(analysis.arc_length_relative_to_cutoff)}

#     data_Yaxis = [ [ {"subplot_title": f"Ray Trajectory, Poloidal Plane",
#                       "subplot_title_fontsize": 16,
#                       "subplot_legend": False,
#                       "subplot_grid": True,
#                       },
#                         {"q_X (m)": np.array(solver_output.q_X)},
#                    ],
#                  ]

#     return



def plot_wavevector(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None):

    metadata = get_metadata(inputs)
    
    data_Xaxis = {"Arc length relative to cut-off (m)": np.array(analysis.arc_length_relative_to_cutoff)}

    data_Yaxis = [ [ {"subplot_title": f"Wavenumber, X-component",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"K_X (m^-1)": np.array(solver_output.K_X)},
                   ],
                   [ {"subplot_title": f"Wavenumber, Y-component",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"K_Y (m^-1)": np.array(solver_output.K_Y)},
                   ],
                   [ {"subplot_title": f"Wavenumber, Z-component",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"K_Z (m^-1)": np.array(solver_output.K_Z)},
                   ],
                   [ {"subplot_title": f"Wavenumber Magnitude",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"K (m^-1)": np.sqrt( np.array(solver_output.K_X)**2 + np.array(solver_output.K_Y)**2 + np.array(solver_output.K_Z)**2) },
                   ],
                 ]
    
    fig, axs = plotter(data_Xaxis, data_Yaxis, metadata, filename,
                       figure_title = r"Individual Wavenumbers",
                       # figure_title_fontsize = 
                       # figure_size = 
                       # metadata_flag = 
                       # metadata_fontsize = 
                       )

    return fig, axs



def plot_widths(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None):

    metadata = get_metadata(inputs)
    
    data_Xaxis = {"Arc length relative to cut-off (m)": np.array(analysis.arc_length_relative_to_cutoff)}

    data_Yaxis = [ [ {"subplot_title": f"Beam Width 1",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"Width 1 (m)": np.array(analysis.beam_width_1)},
                   ],
                   [ {"subplot_title": f"Beam Width 2",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"Width 2 (m)": np.array(analysis.beam_width_2)},
                   ],
                   [ {"subplot_title": f"Beam Width 1 and 2",
                      "subplot_title_fontsize": 16,
                      "subplot_legend": False,
                      "subplot_grid": True,
                      },
                        {"Width 1 (m)": np.array(analysis.beam_width_1)},
                        {"Width 2 (m)": np.array(analysis.beam_width_2)},
                   ],
                 ]
    
    fig, axs = plotter(data_Xaxis, data_Yaxis, metadata, filename,
                       figure_title = r"Beam Widths",
                       # figure_title_fontsize = 
                       # figure_size = 
                       # metadata_flag = 
                       # metadata_fontsize = 
                       )

    return fig, axs