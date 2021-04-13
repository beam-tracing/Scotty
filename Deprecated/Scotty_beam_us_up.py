# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:53:06 2021

@author: VH Chen
"""


@njit(parallel=True)
def beam_us_up(mirror_rotation_angle_scan,
                mirror_tilt_angle,
                launch_freq_GHz_sweep,
                mode_flag,
                vacuumLaunch_flag,
                launch_beam_width,
                launch_beam_radius_of_curvature,
                launch_position,
                find_B_method,
                efit_time_index_scan,
                efit_times,
                vacuum_propagation_flag=False,
                Psi_BC_flag = False,
                input_filename_suffix='',
                plasmaLaunch_K=np.zeros(3),
                plasmaLaunch_Psi_3D_lab_Cartesian=np.zeros([3,3]),
                density_fit_parameters_scan=None
                ):
    """
    Wrapper for beam_me_up
    Easy parallelisation for parameter sweeps
    
    Sweep parameters:
        mirror_rotation_angle_scan
        efit_time_index_scan
        launch_freq_GHz_sweep
        
    Apparently it's generally better to parallelise the outer loop
    I've decided to make that sweep across launch frequencies
    After all, that's the thing I sweep over most frequently
    """
    
    # First prepare some stuff
    # I'm doing it in separate for loops so that later on, when we need to have parallel for loops, things will work
    poloidal_launch_angle_Torbeam_sweep = np.zeros_like(mirror_rotation_angle_scan)
    toroidal_launch_angle_Torbeam_sweep = np.zeros_like(mirror_rotation_angle_scan)
    for kk, mirror_rotation_angle in enumerate(mirror_rotation_angle_scan):            
        toroidal_launch_angle_genray, poloidal_launch_angle_genray = genray_angles_from_mirror_angles(mirror_rotation_angle,mirror_tilt_angle,offset_for_window_norm_to_R = np.rad2deg(math.atan2(125,2432)))
        
        poloidal_launch_angle_Torbeam_sweep[kk] = - poloidal_launch_angle_genray
        toroidal_launch_angle_Torbeam_sweep[kk] = - toroidal_launch_angle_genray    
    
    if mode_flag == 1:
        mode_string = 'O'
    elif mode_flag == -1:
        mode_string = 'X'    

                
    ##
    for ii in prange(len(launch_freq_GHz_sweep)):
        for jj, efit_time_index in enumerate(efit_time_index_scan):
            for kk, mirror_rotation_angle in enumerate(mirror_rotation_angle_scan):        
                beam_me_up( toroidal_launch_angle_Torbeam_sweep[kk],
                            toroidal_launch_angle_Torbeam_sweep[kk],
                            launch_freq_GHz_sweep[ii],
                            mode_flag,
                            vacuumLaunch_flag,
                            launch_beam_width,
                            launch_beam_radius_of_curvature,
                            launch_position,
                            find_B_method,
                            efit_time_index,
                            vacuum_propagation_flag,
                            Psi_BC_flag,
                            poloidal_flux_enter=params_record[ii,2],
                            output_filename_suffix= (
                                                        '_r' #+ f'{mirror_rotation_angle:.1f}'
                                                      # + '_t' + f'{mirror_tilt_angle:.1f}'
                                                      # + '_f' + f'{launch_freq_GHz_sweep[ii]:.1f}'
                                                      # + '_'  + mode_string
                                                      # + '_'  + f'{efit_times[jj]:.3g}' + 'ms'
                                                    ),
                            density_fit_parameters=params_record[ii,:]
                            )    
    
    return None