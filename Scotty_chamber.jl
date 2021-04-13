using Conda
using Distributed
addprocs(4)
@everywhere using PyCall


pushfirst!(PyVector(pyimport("sys")."path"), "")

#Conda.add("matplotlib")

Scotty_beam_me_up = pyimport("Scotty_beam_me_up")

poloidal_launch_angle_Torbeam = 5.0
toroidal_launch_angle_Torbeam = 4.0
launch_freq_GHz = 47.5
mode_flag = -1
vacuumLaunch_flag = true
launch_beam_width = 0.07
launch_beam_radius_of_curvature = -1
launch_position = [2.4,0,0]
find_B_method = "efit"
efit_time_index = 9
vacuum_propagation_flag = true
Psi_BC_flag = true
poloidal_flux_enter = 1.22



# Scotty_beam_me_up.beam_me_up(
#                             poloidal_launch_angle_Torbeam,
#                             toroidal_launch_angle_Torbeam,
#                             launch_freq_GHz,
#                             mode_flag,
#                             vacuumLaunch_flag,
#                             launch_beam_width,
#                             launch_beam_radius_of_curvature,
#                             launch_position,
#                             find_B_method,
#                             efit_time_index,
#                             vacuum_propagation_flag,
#                             Psi_BC_flag,
#                             poloidal_flux_enter,
#                             output_filename_suffix="",
#                             figure_flag= false,
#                             density_fit_parameters=[3.5,-2.1,1.22]
#                             )

# Threads.@threads for efit_time_index in 1:8
#     #output_filename_suffix = str
#     # println(ii)
#     Scotty_beam_me_up.beam_me_up(
#         poloidal_launch_angle_Torbeam,
#         toroidal_launch_angle_Torbeam,
#         launch_freq_GHz,
#         mode_flag,
#         vacuumLaunch_flag,
#         launch_beam_width,
#         launch_beam_radius_of_curvature,
#         launch_position,
#         find_B_method,
#         efit_time_index,
#         vacuum_propagation_flag,
#         Psi_BC_flag,
#         poloidal_flux_enter,
#         output_filename_suffix=string(efit_time_index),
#         figure_flag= false,
#         density_fit_parameters=[3.5,-2.1,1.22]
#         )
# end

np = pyimport("numpy")


@everywhere function pyfoo()
    py"1+1"
end


pmap(x->pyfoo(), 1:2)
