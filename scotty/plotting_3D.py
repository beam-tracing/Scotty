import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scotty.typing import PathLike
from typing import Optional
import xarray as xr

def maybe_make_axis(ax: Optional[plt.Axes], *args, **kwargs) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(*args, **kwargs)
    return ax

def plot_dispersion_relation(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
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
        analysis.arc_length_relative_to_cutoff,
        abs(analysis.H_eigvals.sel(col="X")),
        color="pink",
        linestyle="-",
        linewidth=4,
        label="H_eigvals_1 (X?)",
    )
    ax.plot(
        analysis.arc_length_relative_to_cutoff,
        abs(analysis.H_eigvals.sel(col="Y")),
        color="lightgreen",
        linestyle="-",
        linewidth=4,
        label="H_eigvals_2 (O?)",
    )
    ax.plot(
        analysis.arc_length_relative_to_cutoff,
        abs(analysis.H_eigvals.sel(col="Z")),
        color="gray",
        linestyle="-",
        linewidth=4,
        label="H_eigvals_3",
    )

    ax.plot(
        analysis.arc_length_relative_to_cutoff,
        abs(analysis.H_1_Cardano),
        color="k",
        linestyle=":",
        linewidth=4,
        label="H_1_Cardano",
    )
    ax.plot(
        analysis.arc_length_relative_to_cutoff,
        abs(analysis.H_2_Cardano),
        color="r",
        linestyle=":",
        linewidth=4,
        label="H_2_Cardano (X?)",
    )
    ax.plot(
        analysis.arc_length_relative_to_cutoff,
        abs(analysis.H_3_Cardano),
        color="g",
        linestyle=":",
        linewidth=4,
        label="H_3_Cardano (O?)",
    )

    data_misc = {
        "poloidal_launch_angle_Torbeam": np.array(inputs.poloidal_launch_angle_Torbeam),
        "toroidal_launch_angle_Torbeam": np.array(inputs.toroidal_launch_angle_Torbeam),
    }

    ax.set_title(f'Dispersion relation for \n pol. angle {data_misc["poloidal_launch_angle_Torbeam"]}, tor. angle {data_misc["toroidal_launch_angle_Torbeam"]}')
    ax.legend()

    if filename:
        plt.savefig(f"{filename}")
    
    plt.close()

    return



def plot_trajectory(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    
    data_misc = {
        "poloidal_launch_angle_Torbeam": np.array(inputs.poloidal_launch_angle_Torbeam),
        "toroidal_launch_angle_Torbeam": np.array(inputs.toroidal_launch_angle_Torbeam),
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
    ax.scatter(data_Yaxis["q_X"][0], data_Yaxis["q_Y"][0], data_Yaxis["q_Z"][0], color='black', label="Start Point")
    ax.plot(data_Yaxis["q_X"], data_Yaxis["q_Y"], data_Yaxis["q_Z"], linestyle='-', linewidth=2, zorder=1, label="Trajectory")
    ax.set_ylim(-0.05, 0.05)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(f'Ray Trajectory') # , pol {data_misc["poloidal_launch_angle_Torbeam"]}, tor {data_misc["toroidal_launch_angle_Torbeam"]}')
    ax.legend()
    plt.draw()
    if filename: plt.savefig(f"{filename}")
    plt.close()



def plot_trajectories_individually(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    
    data_misc = {
        "poloidal_launch_angle_Torbeam": np.array(inputs.poloidal_launch_angle_Torbeam),
        "toroidal_launch_angle_Torbeam": np.array(inputs.toroidal_launch_angle_Torbeam),
    }

    data_Xaxis = {
        "arc_length_relative_to_cutoff": np.array(analysis.arc_length_relative_to_cutoff),
    }
    
    data_Yaxis = {
        "q_X": np.array(solver_output.q_X),
        "q_Y": np.array(solver_output.q_Y),
        "q_Z": np.array(solver_output.q_Z),
    }

    # Plotting ray trajectory one by one w.r.t. to arc length
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs = axs.flatten()
    counter = 0
    for key in ["q_X", "q_Y", "q_Z"]:
        axs[counter].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[key], linestyle='-', linewidth=2, zorder=1)
        axs[counter].set_xlabel("Arc length from cutoff (m)")
        axs[counter].set_title(f'{key}, pol {data_misc["poloidal_launch_angle_Torbeam"]}, tor {data_misc["toroidal_launch_angle_Torbeam"]}')
        axs[counter].legend(loc="best")
        axs[counter].grid(True, which="both")
        counter += 1
    
    axs[0].set_title("Plot of Ray Trajectory, X coordinate against arc length") # TO REMOVE -- for FYP graph
    axs[0].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    axs[0].set_ylabel("X coordinate (m)") # TO REMOVE -- for FYP graph

    axs[1].set_title("Plot of Ray Trajectory, Y coordinate against arc length") # TO REMOVE -- for FYP graph
    axs[1].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    axs[1].set_ylabel("Y coordinate (m)") # TO REMOVE -- for FYP graph

    axs[2].set_title("Plot of Ray Trajectory, Z coordinate against arc length") # TO REMOVE -- for FYP graph
    axs[2].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    axs[2].set_ylabel("Z coordinate (m)") # TO REMOVE -- for FYP graph

    plt.tight_layout()
    plt.draw()
    if filename: plt.savefig(f"{filename}")
    plt.close()

    return




def plot_wavevector(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    
    data_misc = {
        "poloidal_launch_angle_Torbeam": np.array(inputs.poloidal_launch_angle_Torbeam),
        "toroidal_launch_angle_Torbeam": np.array(inputs.toroidal_launch_angle_Torbeam),
    }

    data_Xaxis = {
        "arc_length_relative_to_cutoff": np.array(analysis.arc_length_relative_to_cutoff),
    }
    
    data_Yaxis = {
        "K_X": np.array(solver_output.K_X),
        "K_Y": np.array(solver_output.K_Y),
        "K_Z": np.array(solver_output.K_Z),
    }
    data_Yaxis["K_magnitude"] = np.sqrt(data_Yaxis["K_X"]**2 + data_Yaxis["K_Y"]**2 + data_Yaxis["K_Z"]**2)

    # Plotting wavevectors one by one w.r.t. to arc length
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    axs = axs.flatten()
    counter = 0
    for key in ["K_magnitude", "K_X", "K_Y", "K_Z"]:
        axs[counter].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[key], linestyle='-', linewidth=2, zorder=1)
        axs[counter].set_xlabel("Arc length from cutoff (m)")
        axs[counter].set_title(f'{key}, pol {data_misc["poloidal_launch_angle_Torbeam"]}, tor {data_misc["toroidal_launch_angle_Torbeam"]}')
        axs[counter].legend(loc="best")
        axs[counter].grid(True, which="both")
        counter += 1
    
    axs[0].set_title(r"Plot of Wavenumber K against arc length") # TO REMOVE -- for FYP graph
    axs[0].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    axs[0].set_ylabel(r"Wavenumber K ($m^{-1}$)") # TO REMOVE -- for FYP graph

    axs[1].set_title(r"Plot of Wavenumber X-component, $K_X$ against arc length") # TO REMOVE -- for FYP graph
    axs[1].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    axs[1].set_ylabel(r"Wavenumber X-component, $K_X$ ($m^{-1}$)") # TO REMOVE -- for FYP graph

    axs[2].set_title(r"Plot of Wavenumber X-component, $K_Y$ against arc length") # TO REMOVE -- for FYP graph
    axs[2].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    axs[2].set_ylabel(r"Wavenumber X-component, $K_Y$ ($m^{-1}$)") # TO REMOVE -- for FYP graph

    axs[3].set_title(r"Plot of Wavenumber X-component, $K_Z$ against arc length") # TO REMOVE -- for FYP graph
    axs[3].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    axs[3].set_ylabel(r"Wavenumber X-component, $K_Z$ ($m^{-1}$)") # TO REMOVE -- for FYP graph

    plt.tight_layout()
    plt.draw()
    if filename: plt.savefig(f"{filename}")
    plt.close()

    return



def plot_localisations(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    
    data_misc = {
        "poloidal_launch_angle_Torbeam": np.round(np.array(inputs.poloidal_launch_angle_Torbeam), 1),
        "toroidal_launch_angle_Torbeam": np.round(np.array(inputs.toroidal_launch_angle_Torbeam), 1),
    }
    
    data_Xaxis = {
        "arc_length_relative_to_cutoff": np.array(analysis.arc_length_relative_to_cutoff),
    }
    
    data_Yaxis = {
        "loc_p":   np.array(analysis.loc_p),
        "loc_r":   np.array(analysis.loc_r),
        "loc_b":   np.array(analysis.loc_b),
        "loc_m":   np.array(analysis.loc_m),
        "loc_s_10_3":   np.array(analysis.loc_s_10_3),
        "loc_all_10_3": np.array(analysis.loc_all_10_3),
        "loc_s_13_3":   np.array(analysis.loc_s_13_3),
        "loc_all_13_3": np.array(analysis.loc_all_13_3),
    }
    
    # Plotting all localisations against l_lc
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    for i, quantity in enumerate(data_Yaxis.keys()):
        if i in [0,1,2,3]:
            axes[i].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], label="Both power laws", linewidth=3)
            axes[i].set_title(quantity, fontsize=20)
            # axes[i].set_xlabel("Arc length from cutoff (m)", fontsize=20, labelpad=30)
            # axes[i].legend(fontsize=12)
            axes[i].tick_params(axis="both", which="major", labelsize="30")
        elif i in [4]:
            axes[4].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="blue", label="-10/3", linewidth=3)
            axes[4].set_title(quantity, fontsize=20)
            # axes[4].set_xlabel("Arc length from cutoff (m)", fontsize=20, labelpad=30)
            axes[4].legend(fontsize=20)
            axes[4].tick_params(axis="both", which="major", labelsize="30")
        elif i in [6]:
            axes[4].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="red", label="-13/3", linewidth=3)
            axes[4].set_title(quantity, fontsize=20)
            # axes[4].set_xlabel("Arc length from cutoff (m)", fontsize=20, labelpad=30)
            axes[4].legend(fontsize=20)
            axes[4].tick_params(axis="both", which="major", labelsize="30")

        # elif i in [4]:
        #     axes[i].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="blue", label="Power law -10/3")
        #     axes[i].set_title("loc_s")
        #     axes[i].set_xlabel("Arc length from cutoff (m)")
        # elif i in [5]:
        #     axes[i].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="blue", label="Power law -10/3")
        #     axes[i].set_title("loc_all")
        #     axes[i].set_xlabel("Arc length from cutoff (m)")
        # elif i in [6,7]:
        #     axes[i-2].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="red", label="Power law -13/3")
        #     axes[i-2].legend()
    
    axes[0].set_title("Polarisation piece", fontsize=30, pad=30) # TO REMOVE -- for FYP graph
    # axes[0].set_xlabel("Arc length from cutoff (m)", fontsize=30, labelpad=30) # TO REMOVE -- for FYP graph
    axes[0].set_ylabel("Localisations (arb. units)", fontsize=30, labelpad=30) # TO REMOVE -- for FYP graph

    axes[1].set_title("Ray piece", fontsize=30, pad=30) # TO REMOVE -- for FYP graph
    # axes[1].set_xlabel("Arc length from cutoff (m)", fontsize=30, labelpad=30) # TO REMOVE -- for FYP graph
    # axes[1].set_ylabel("Localisation (arb. units)", fontsize=16, labelpad=30) # TO REMOVE -- for FYP graph

    axes[2].set_title("Beam piece", fontsize=30, pad=30) # TO REMOVE -- for FYP graph
    axes[2].set_xlabel("Arc length from cutoff (m)", fontsize=30, labelpad=30) # TO REMOVE -- for FYP graph
    # axes[2].set_ylabel("Localisation (arb. units)", fontsize=16, labelpad=30) # TO REMOVE -- for FYP graph

    axes[3].set_title("Mismatch piece", fontsize=30, pad=30) # TO REMOVE -- for FYP graph
    # axes[3].set_xlabel("Arc length from cutoff (m)", fontsize=30, labelpad=30) # TO REMOVE -- for FYP graph
    # axes[3].set_ylabel("Localisation (arb. units)", fontsize=16, labelpad=30) # TO REMOVE -- for FYP graph

    axes[4].set_title("Spectrum piece", fontsize=30, pad=30) # TO REMOVE -- for FYP graph
    # axes[4].set_xlabel("Arc length from cutoff (m)", fontsize=30, labelpad=30) # TO REMOVE -- for FYP graph
    # axes[4].set_ylabel("Localisation (arb. units)", fontsize=16, labelpad=30) # TO REMOVE -- for FYP graph

    plt.tight_layout()
    plt.draw()
    # filename = fr"C:\Users\eduar\OneDrive\OneDrive - EWIKARTA001\OneDrive - Nanyang Technological University\University\2. To Be Uploaded\Year 4, PH4421 Final Year Project\Mid-Term Report\Figures\G. Results\3. stellarator\pol{round(data_misc['poloidal_launch_angle_Torbeam']*10)}\indiv loc, power law"
    filename = fr"C:\Users\eduar\OneDrive\OneDrive - EWIKARTA001\OneDrive - Nanyang Technological University\! TEMPORARY\pol{round(data_misc['poloidal_launch_angle_Torbeam']*10)}_tor{round(data_misc['toroidal_launch_angle_Torbeam']*10)}\G. stellarator - pol{round(data_misc['poloidal_launch_angle_Torbeam'])}_tor{round(data_misc['toroidal_launch_angle_Torbeam'], 1)} indiv loc.png"
    plt.savefig(filename)





    # fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # for i, quantity in enumerate(data_Yaxis.keys()):
    #     if i in [0,1,2,3]:
    #         axes[i].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="red", label="Power law -13/3")
    #         axes[i].set_title(quantity)
    #         axes[i].set_xlabel("Arc length from cutoff (m)")
    #         axes[i].legend()
    #     elif i in [6]:
    #         axes[4].plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="red", label="Power law -13/3")
    #         axes[4].set_title(quantity)
    #         axes[4].set_xlabel("Arc length from cutoff (m)")
    #         axes[4].legend()

    # axes[0].set_title("Polarisation piece") # TO REMOVE -- for FYP graph
    # axes[0].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    # axes[0].set_ylabel("Localisation (arb. units)") # TO REMOVE -- for FYP graph

    # axes[1].set_title("Ray piece") # TO REMOVE -- for FYP graph
    # axes[1].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    # axes[1].set_ylabel("Localisation (arb. units)") # TO REMOVE -- for FYP graph

    # axes[2].set_title("Beam piece") # TO REMOVE -- for FYP graph
    # axes[2].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    # axes[2].set_ylabel("Localisation (arb. units)") # TO REMOVE -- for FYP graph

    # axes[3].set_title("Mismatch piece") # TO REMOVE -- for FYP graph
    # axes[3].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    # axes[3].set_ylabel("Localisation (arb. units)") # TO REMOVE -- for FYP graph

    # axes[4].set_title("Spectrum piece, power law -13/3") # TO REMOVE -- for FYP graph
    # axes[4].set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    # axes[4].set_ylabel("Localisation (arb. units)") # TO REMOVE -- for FYP graph

    # plt.tight_layout()
    # plt.draw()
    # filename = fr"C:\Users\eduar\OneDrive\OneDrive - EWIKARTA001\OneDrive - Nanyang Technological University\University\2. To Be Uploaded\Year 4, PH4421 Final Year Project\Mid-Term Report\Figures\G. Results\3. stellarator\pol{round(data_misc['poloidal_launch_angle_Torbeam']*10)}\indiv loc, power law -13_3"
    # plt.savefig(filename)





    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    for i, quantity in enumerate(data_Yaxis.keys()):
        if i in [5]:
            axes.plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="blue", label="Power law -10/3", linewidth=2)
            axes.set_title(quantity, fontsize=20)
            axes.set_xlabel("Arc length from cutoff (m)", fontsize=20, labelpad=20)
            axes.legend(fontsize=12)
            axes.tick_params(axis="both", which="major", labelsize="20")
        if i in [7]:
            axes.plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="red", label="Power law -13/3", linewidth=2)
            axes.set_title(quantity, fontsize=20)
            axes.set_xlabel("Arc length from cutoff (m)", fontsize=20, labelpad=20)
            axes.legend(fontsize=12)
            axes.tick_params(axis="both", which="major", labelsize="20")

    axes.set_title(fr"Overall Localisation" "\n" fr"$\alpha_{{pol}}$={round(data_misc['poloidal_launch_angle_Torbeam'])}$^\circ$, $\alpha_{{tor}}$={round(data_misc['toroidal_launch_angle_Torbeam'], 1)}$^\circ$", fontsize=20, pad=20) # TO REMOVE -- for FYP graph
    axes.set_xlabel("Arc length from cutoff (m)", fontsize=20, labelpad=20) # TO REMOVE -- for FYP graph
    axes.set_ylabel("Localisation (arb. units)", fontsize=20, labelpad=20) # TO REMOVE -- for FYP graph

    plt.tight_layout()
    plt.draw()
    # filename = fr"C:\Users\eduar\OneDrive\OneDrive - EWIKARTA001\OneDrive - Nanyang Technological University\University\2. To Be Uploaded\Year 4, PH4421 Final Year Project\Mid-Term Report\Figures\G. Results\3. stellarator\pol{round(data_misc['poloidal_launch_angle_Torbeam']*10)}\overall loc, both power law"
    filename = fr"C:\Users\eduar\OneDrive\OneDrive - EWIKARTA001\OneDrive - Nanyang Technological University\! TEMPORARY\pol{round(data_misc['poloidal_launch_angle_Torbeam']*10)}_tor{round(data_misc['toroidal_launch_angle_Torbeam']*10)}\G. stellarator - pol{round(data_misc['poloidal_launch_angle_Torbeam'])}_tor{round(data_misc['toroidal_launch_angle_Torbeam'], 1)} cum loc.png"
    plt.savefig(filename)





    # fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    # for i, quantity in enumerate(data_Yaxis.keys()):
    #     if i in [7]:
    #         axes.plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis[quantity], color="red", label="Power law -13/3")
    #         axes.set_title(quantity)
    #         axes.set_xlabel("Arc length from cutoff (m)")
    #         axes.legend()

    # axes.set_title("Overall Localisation, power law -13/3") # TO REMOVE -- for FYP graph
    # axes.set_xlabel("Arc length from cutoff (m)") # TO REMOVE -- for FYP graph
    # axes.set_ylabel("Localisation (arb. units)") # TO REMOVE -- for FYP graph

    # plt.tight_layout()
    # plt.draw()
    # filename = fr"C:\Users\eduar\OneDrive\OneDrive - EWIKARTA001\OneDrive - Nanyang Technological University\University\2. To Be Uploaded\Year 4, PH4421 Final Year Project\Mid-Term Report\Figures\G. Results\3. stellarator\pol{round(data_misc['poloidal_launch_angle_Torbeam']*10)}\overall loc, power law -13_3"
    # plt.savefig(filename)





    # fig.suptitle(f'Localisation graphs for \n pol. angle {data_misc["poloidal_launch_angle_Torbeam"]}, tor. angle {data_misc["toroidal_launch_angle_Torbeam"]}')
    # plt.tight_layout()
    # plt.draw()

    # if filename:
    #     plt.savefig(f"{filename}")

    # plt.close()

    return



def plot_theta_m(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    
    data_misc = {
        "poloidal_launch_angle_Torbeam": np.array(inputs.poloidal_launch_angle_Torbeam),
        "toroidal_launch_angle_Torbeam": np.array(inputs.toroidal_launch_angle_Torbeam),
    }
    
    data_Xaxis = {
        "arc_length_relative_to_cutoff": np.array(analysis.arc_length_relative_to_cutoff),
    }
    
    data_Yaxis = {
        "theta_m": np.array(analysis.theta_m),
    }
    
    # Plot theta_m
    fig, ax = plt.subplots()
    ax.plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis["theta_m"], linestyle='-', linewidth=2, zorder=1, label="theta_m")
    ax.set_xlabel("Arc length from cutoff (m)")
    ax.set_title(f'theta_m, pol {data_misc["poloidal_launch_angle_Torbeam"]}, tor {data_misc["toroidal_launch_angle_Torbeam"]}')
    ax.legend()
    plt.draw()
    if filename: plt.savefig(f"{filename}")
    plt.close()

    return



def plot_delta_theta_m(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    
    data_misc = {
        "poloidal_launch_angle_Torbeam": np.array(inputs.poloidal_launch_angle_Torbeam),
        "toroidal_launch_angle_Torbeam": np.array(inputs.toroidal_launch_angle_Torbeam),
    }
    
    data_Xaxis = {
        "arc_length_relative_to_cutoff": np.array(analysis.arc_length_relative_to_cutoff),
    }
    
    data_Yaxis = {
        "delta_theta_m": np.array(analysis.delta_theta_m),
    }
    
    # Plot delta_theta_m
    fig, ax = plt.subplots()
    ax.plot(data_Xaxis["arc_length_relative_to_cutoff"], data_Yaxis["delta_theta_m"], linestyle='-', linewidth=2, zorder=1, label="delta_theta_m")
    ax.set_xlabel("Arc length from cutoff (m)")
    ax.set_title(f'delta_theta_m, pol {data_misc["poloidal_launch_angle_Torbeam"]}, tor {data_misc["toroidal_launch_angle_Torbeam"]}')
    ax.legend()
    plt.draw()
    if filename: plt.savefig(f"{filename}")
    plt.close()

    return