import datatree as dt
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
from scotty.beam_me_up import beam_me_up
from scotty.beam_me_up_3D_temp import beam_me_up_3D
from scotty.fun_general import cylindrical_to_cartesian, find_K_lab_Cartesian
from scotty.fun_general_3D import round_to_1sf, get_arr_bounds, find_beam_widths_curvs
from scotty.typing import FloatArray

log = logging.getLogger(__name__)

def benchmark_me_up_3D(beam_me_up_kwargs: dict, beam_me_up_3D_kwargs: dict):

    # Defining stuff here for ease
    _pad = 20

    # Pre-processing the dictionaries
    if "create_magnetic_geometry_3D" not in beam_me_up_3D_kwargs:
        beam_me_up_3D_kwargs["create_magnetic_geometry_3D"] = {}

    (cyl_dt,
     cyl_field,
     cyl_ne,
     cyl_Te,
     cyl_H) = beam_me_up(**beam_me_up_kwargs)
    
    # Converting the ray trajectory from cylindrical to cartesian
    # to obtain bounding box for 3d field
    cyl_q_R    = np.array(cyl_dt["solver_output"]["q_R"])
    cyl_q_zeta = np.array(cyl_dt["solver_output"]["q_zeta"])
    cyl_q_Z    = np.array(cyl_dt["solver_output"]["q_Z"])
    cyl_q_X = cyl_q_R * np.cos(cyl_q_zeta)
    cyl_q_Y = cyl_q_R * np.sin(cyl_q_zeta)

    # Getting the keyword arguments for the 3d field
    R_coords = cyl_field.R_coord
    Z_coords = cyl_field.Z_coord
    cyl_R_spacing = abs(round_to_1sf(R_coords[1] - R_coords[0]))
    cyl_Z_spacing = abs(round_to_1sf(Z_coords[1] - Z_coords[0]))
    cart_X_spacing = beam_me_up_3D_kwargs["create_magnetic_geometry_3D"].pop("X_spacing", cyl_R_spacing)
    cart_Y_spacing = beam_me_up_3D_kwargs["create_magnetic_geometry_3D"].pop("Y_spacing", cyl_R_spacing)
    cart_Z_spacing = beam_me_up_3D_kwargs["create_magnetic_geometry_3D"].pop("Z_spacing", cyl_Z_spacing)
    pad = beam_me_up_3D_kwargs["create_magnetic_geometry_3D"].pop("pad", _pad)

    # Finding and constructing the bounding box ranges by padding the
    # ranges from before with 10 extra points on all sides because
    # interpolation requires at least 6 more points to be padded
    # and then saving them as entries of a kwarg dictionary. We also
    # check to make sure the padded points do not exit the bounds of
    # the original field
    def find_padded_array(arr: FloatArray, userpad: int, spacing: float, coord_arr: FloatArray = None) -> np.ndarray:
        _pad = max(userpad, pad)
        _min_coord = min(arr) - spacing*_pad
        _max_coord = max(arr) + spacing*_pad
        if coord_arr is not None: return np.array([x for x in coord_arr if _min_coord <= x <= _max_coord])
        else:                     return np.linspace(*get_arr_bounds(_min_coord, _max_coord, spacing))
    
    if "X" not in beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]:
        _temp = find_padded_array(cyl_q_X, pad, cart_X_spacing, R_coords)
        beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]["X"] = _temp
        log.debug(f"Creating X-coordinates: {_temp}")
    
    if "Y" not in beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]:
        _temp = find_padded_array(cyl_q_Y, pad, cart_Y_spacing)
        beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]["Y"] = _temp
        log.debug(f"Creating Y-coordinates: {_temp}")
    
    if "Z" not in beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]:
        _temp = find_padded_array(cyl_q_Z, pad, cart_Z_spacing, Z_coords)
        beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]["Z"] = _temp
        log.debug(f"Creating Z-coordinates: {_temp}")
    
    beam_me_up_3D_kwargs["find_B_method"] = cyl_field

    (cart_dt,
     cart_field,
     cart_ne,
     cart_Te,
     cart_H) = beam_me_up_3D(**beam_me_up_3D_kwargs)

    return (cyl_dt, cyl_field, cyl_ne, cyl_Te, cyl_H), (cart_dt, cart_field, cart_ne, cart_Te, cart_H)



def benchmark_plots(beam_me_up_dt: dt.DataTree, beam_me_up_3D_dt: dt.DataTree):

    # Get all of the data first
    def _convert_cyld_to_cart_data():
        q = np.array([
            beam_me_up_dt["solver_output"]["q_R"],
            beam_me_up_dt["solver_output"]["q_zeta"],
            beam_me_up_dt["solver_output"]["q_Z"]])
        q_cart = np.array(cylindrical_to_cartesian(*q))

        K = np.array([
            beam_me_up_dt["solver_output"]["K_R"],
            np.full_like(beam_me_up_dt["solver_output"]["K_R"], beam_me_up_dt["inputs"]["launch_K"][1]),
            beam_me_up_dt["solver_output"]["K_Z"]])
        K_cart = find_K_lab_Cartesian(K, q)

        Psi_w = np.zeros((len(q[0]), 2,2), dtype=np.complex128)
        Psi_w[:, 0, 0] = beam_me_up_dt["analysis"]["Psi_xx"]
        Psi_w[:, 1, 0] = beam_me_up_dt["analysis"]["Psi_xy"]
        Psi_w[:, 0, 1] = beam_me_up_dt["analysis"]["Psi_xy"]
        Psi_w[:, 1, 1] = beam_me_up_dt["analysis"]["Psi_yy"]

        curv1, curv2, width1, width2 = find_beam_widths_curvs(
            Psi_w, K_cart.T, np.array(beam_me_up_dt["analysis"]["g_hat_Cartesian"])
        )

        d = {
            "Arc length [m]": np.array(beam_me_up_dt["analysis"]["arc_length"]),
            "Arc length relative to cutoff [m]": np.array(beam_me_up_dt["analysis"]["arc_length_relative_to_cutoff"]),
            "q_X": q_cart[0],
            "q_Y": q_cart[1],
            "q_Z": q_cart[2],
            "K_X": K_cart[0],
            "K_Y": K_cart[1],
            "K_Z": K_cart[2],
            "Principal Width 1": width1,
            "Principal Width 2": width2,
            "Principal Curvature 1": curv1,
            "Principal Curvature 2": curv2,
            "H_Cardano": np.array(beam_me_up_dt["analysis"]["H_Cardano"]),
        }
        
        return d

    cyld_d = _convert_cyld_to_cart_data()

    cart_d = {
        "Arc length [m]": np.array(beam_me_up_3D_dt["analysis"]["arc_length"]),
        "Arc length relative to cutoff [m]": np.array(beam_me_up_3D_dt["analysis"]["arc_length_relative_to_cutoff"]),
        "q_X":    np.array(beam_me_up_3D_dt["solver_output"]["q_X"]),
        "q_Y":    np.array(beam_me_up_3D_dt["solver_output"]["q_Y"]),
        "q_Z":    np.array(beam_me_up_3D_dt["solver_output"]["q_Z"]),
        "K_X":    np.array(beam_me_up_3D_dt["solver_output"]["K_X"]),
        "K_Y":    np.array(beam_me_up_3D_dt["solver_output"]["K_Y"]),
        "K_Z":    np.array(beam_me_up_3D_dt["solver_output"]["K_Z"]),
        "Principal Width 1": np.array(beam_me_up_3D_dt["analysis"]["beam_width_1"]),
        "Principal Width 2": np.array(beam_me_up_3D_dt["analysis"]["beam_width_2"]),
        "Principal Curvature 1": np.array(beam_me_up_3D_dt["analysis"]["beam_curvature_1"]),
        "Principal Curvature 2": np.array(beam_me_up_3D_dt["analysis"]["beam_curvature_2"]),
        "H_Cardano": np.array(beam_me_up_3D_dt["analysis"]["H_Cardano"]),
    }

    _q_launch = np.round(np.array(beam_me_up_3D_dt["inputs"]["launch_position_cartesian"]), 5)
    _pol_angle =    round(float(beam_me_up_3D_dt["inputs"]["poloidal_launch_angle_Torbeam"]), 5)
    _tor_angle =    round(float(beam_me_up_3D_dt["inputs"]["toroidal_launch_angle_Torbeam"]), 5)
    _launch_freq =  round(float(beam_me_up_3D_dt["inputs"]["launch_freq_GHz"]), 5)
    _launch_width = round(float(beam_me_up_3D_dt["inputs"]["launch_beam_width"]), 5)
    _launch_curv =  round(float(beam_me_up_3D_dt["inputs"]["launch_beam_curvature"]), 5)

    # Making the plots now
    fig, axs = plt.subplots(5, 3, figsize=(15, 20))

    cyld_color = "blue"
    cyld_linestyle = ":"
    cyld_linewidth = 3
    cart_color = "red"
    cart_linestyle = "-"
    cart_linewidth = 3

    for i, k in enumerate(cart_d):
        if   k == "Principal Width 1":     n = 2, 0
        elif k == "Principal Width 2":     n = 2, 1
        elif k == "Principal Curvature 1": n = 3, 0
        elif k == "Principal Curvature 2": n = 3, 1
        elif k == "H_Cardano":             n = 4, 0
        elif k == "Arc length relative to cutoff [m]":
            n = 4, 2
            axs[n].plot([-1,-1], color=cart_color, label="Cart", linewidth=cart_linewidth, linestyle=cart_linestyle)
            axs[n].plot([1,1], color=cyld_color, label="Cyld", linewidth=cyld_linewidth, linestyle=cyld_linestyle)
            axs[n].legend(loc="upper center", framealpha=1, prop={"size": 35})
            txtstr = "\n".join((
                fr"          Arc length [m]",
                fr"",
                fr"",
                fr"Launch position (X,Y,Z) =",
                fr"   {list(_q_launch)}",
                fr"",
                fr"Poloidal launch angle = {_pol_angle}$^o$",
                fr"",
                fr"Toroidal launch angle = {_tor_angle}$^o$",
                fr"",
                fr"Frequency = {_launch_freq} GHz",
                fr"",
                fr"Width = {_launch_width} m",
                fr"",
                fr"Curvature = {_launch_curv} m$^{{-1}}$"
            ))
            axs[n].text(-0.2, 2.1, txtstr, fontsize=20, verticalalignment="top")
            axs[n].set_axis_off()  
            axs[n].set_xlim(1,2)
            axs[n].set_ylim(1,2)
        else: n = math.floor((i-2) / 3), (i-2) - math.floor((i-2) / 3)*3

        if "Arc length" not in k:
            axs[n].plot(cart_d["Arc length [m]"], cart_d[k], color=cart_color, linewidth=cart_linewidth, linestyle=cart_linestyle)
            axs[n].plot(cyld_d["Arc length [m]"], cyld_d[k], color=cyld_color, linewidth=cyld_linewidth, linestyle=cyld_linestyle)
            axs[n].set_title(f"{str(k)}")
        
        if "Width" in k:
            axs[2,2].plot(cart_d["Arc length [m]"], cart_d[k], color=cart_color, linewidth=cart_linewidth, linestyle=cart_linestyle)
            axs[2,2].plot(cyld_d["Arc length [m]"], cyld_d[k], color=cyld_color, linewidth=cyld_linewidth, linestyle=cyld_linestyle)
            axs[2,2].set_title(f"Principal Widths")
            axs[2,2].grid(True)
        
        if "Curvature" in k:
            axs[3,2].plot(cart_d["Arc length [m]"], cart_d[k], color=cart_color, linewidth=cart_linewidth, linestyle=cart_linestyle)
            axs[3,2].plot(cyld_d["Arc length [m]"], cyld_d[k], color=cyld_color, linewidth=cyld_linewidth, linestyle=cyld_linestyle)
            axs[3,2].set_title(f"Principal Widths")
            axs[3,2].grid(True)
        
        if k == "H_Cardano":
            axs[n].set_ylim(-0.01, 0.01)
        
        axs[4,1].axis("off")
        axs[4,2].axis("off")
        axs[n].grid(True)
    
    plt.suptitle(f"Benchmarking Plots", size=25, va="top", y=0.92)
    plt.subplots_adjust(hspace=0.25)
    plt.show()
    
    return fig, axs