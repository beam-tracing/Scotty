
from scotty.beam_me_up import beam_me_up
from scotty.fun_general import freq_GHz_to_wavenumber
import numpy as np
import time
import multiprocessing
import itertools
 
# Initialize the kwargs_dict
#K0 = freq_GHz_to_wavenumber(launch_freq_GHz)

  
# Wrapper function to specify fixed arguments and feed iterated arguments into
def wrap_me_up(arg_tuple):  
    # You would want to change this for whatever variables you are sweeping
    frequency = arg_tuple[0] 
    poloidal_angle = arg_tuple[1]
    toroidal_angle = arg_tuple[2]
    scaling = arg_tuple[3]

    # This bit is specific to iterations over equilibrium scalings
    eq_path = "eq_path" 
    home_path = eq_path + scaling # Path to ne, magnetic data, Te data with standard names
    
    kwargs = {
        'mode_flag': -1, # vary this 1/-1, usually 1 is X-mode, -1 is O-mode
        'launch_beam_curvature': 0.0, 
        'launch_position': np.array([ 2.8,  0., -0.15]),
        'find_B_method': 'omfit',
        'equil_time': None,
        'shot': None,
        'Psi_BC_flag': 'discontinuous', 
        'figure_flag': False,
        'vacuum_propagation_flag': True,
        'vacuumLaunch_flag': True,
        'poloidal_flux_enter': 1.0**2,   
        'input_filename_suffix': '',
        'magnetic_data_path': home_path,
        'ne_data_path': home_path,
        'output_path': None,
        'delta_R': 1e-06,
        'delta_Z': -1e-06,
        'delta_K_R': 0.1, 
        'delta_K_zeta': 0.1,
        'delta_K_Z': 0.1,
        'interp_smoothing': 0.0,
        'len_tau': 1002,
        'rtol': 1e-4,
        'atol': 1e-7,
        'output_filename_suffix': '',      
        'density_fit_method': 'smoothing-spline-file',
        'poloidal_flux_zero_density': 1.00000001**2,
        # Temperature parameters
        'relativistic_flag': False,
        'Te_data_path': home_path,
        'temperature_fit_method': 'smoothing-spline-file',
        #'temperature_fit_parameters': np.array([20, ]),
        'poloidal_flux_zero_temperature': 1.00000001**2,
        'output_path': 'output_path' + scaling, # Insert your output path here
        'launch_freq_GHz': frequency,
        'poloidal_launch_angle_Torbeam': poloidal_angle,
        'toroidal_launch_angle_Torbeam': toroidal_angle,       
        'launch_beam_width': np.sqrt(0.5*1/freq_GHz_to_wavenumber(frequency)), 
        'output_filename_suffix': f'_freq{frequency}_pol{poloidal_angle}_tor{toroidal_angle}', # F-string file name format
    }
    

    beam_me_up(**kwargs) # Calls Scotty


# Part that pools arguments for multiprocessing
def parallel_beam_me_up(args_list):
    with multiprocessing.Pool() as pool:
        pool.map(wrap_me_up, args_list)  

# Main execution part
if __name__ == '__main__':
    
    # Define your sweep variables here
    Frequencies = np.arange(150, 191, 2)
    Toroidal_Angles = np.arange(-10, 10, 1)
    Poloidal_Angles = np.arange(-20, 10, 1)
    scalings = ('0.7\\' , '0.8\\', '0.9\\' , '1.0\\' , '1.1\\')

    errorcount = 0

    for poloidal_angle in Poloidal_Angles:
        Poloidal_Angle = (poloidal_angle,)
        # Code that forces multiprocessing to pool poloidal angle arguments in order for debugging purposes
        # Otherwise directly put all sweep variables into the itertools.product function; all variables
        # will be pooled in random order

        args_list = list(itertools.product(Frequencies, Poloidal_Angle, Toroidal_Angles, scalings))
        try:
            parallel_beam_me_up(args_list)
        except ValueError: # Catch situations where Scotty does not run to completion
            errorcount += 1
            print('current error count is:', errorcount)
            continue
    
    print('total error count is', errorcount)
    


