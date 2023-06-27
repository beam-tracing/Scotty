"""Functions to load poloidal + toroidal + frequency sweeps of 
Scotty output data. Uses xarrays with poloidal angle, toroidal
angle and frequency as axes. 

"""

# Copyright 2023, Valerian Hall-Chen and Scotty contributors
# SPDX-License-Identifier: GPL-3.0


import xarray as xr
import numpy as np

class SweepDataset:
    """Stores an xarray DataSet of Scotty output data indexed by frequency, 
    poloidal angle and toroidal angle. Contains class methods for analysing
    and appending onto the dataset. Includes a method to get the stored
    DataSet to operate on directly.

    Args:
                input_path (string): path to home folder containing all outputs

                Frequencies (iterable): Array or List of indexing frequencies

                Toroidal_Angles (iterable): Array or List of indexing toroidal launch angles

                Poloidal_Angles (iterable): Array or List of indexing poloidal launch angles

                filepath_format (f-string): an f-string that specifies the path string to be
                appended to the end of the input_path to access outputs of individual runs.

                Note that the file path or filenames must include {frequency}, {toroidal_angle},
                and {poloidal_angle} specified in Frequencies, Toroidal_Angles, and Poloidal_Angles,
                as well as a {output_type} corresponding to the type of scotty output (analysis_output,
                data_input, data_output, solver_output).

    Output: 
        None
    """

    def __init__(
    self,
    input_path,
    Frequencies,
    Toroidal_Angles,
    Poloidal_Angles,
    filepath_format,
    Output_Types = ('analysis_output', 'data_input', 'data_output', 'solver_output'),
):

        ds = xr.Dataset()
        variables = [
            'delta_theta_m', 
            'theta_m_output', 
            'theta_output', 
            'cutoff_index', 
            'q_R_array',
            'q_Z_array',
            'q_zeta_array',
            'K_magnitude_array',
            'K_R_array',
            'K_Z_array',
            'K_zeta_initial',
            'poloidal_flux_output',
            ]
        
        for variable in variables:
            ds[variable] = xr.DataArray(
                coords=[
                    ("frequency", Frequencies),
                    ("toroidal_angle", Toroidal_Angles),
                    ("poloidal_angle", Poloidal_Angles)
                ])
        print(ds)
        
        for frequency in Frequencies:
            for toroidal_angle in Toroidal_Angles:
                for poloidal_angle in Poloidal_Angles:
                    index = {
                    'frequency': frequency, 
                    'toroidal_angle': toroidal_angle, 
                    'poloidal_angle': poloidal_angle}

                    # File paths for each type of scotty output file
                    path_analysis = input_path + filepath_format.format(
                        frequency=frequency, 
                        toroidal_angle=toroidal_angle, 
                        poloidal_angle=poloidal_angle,
                        output_type= Output_Types[0])

                    path_input = input_path + filepath_format.format(
                        frequency=frequency, 
                        toroidal_angle=toroidal_angle, 
                        poloidal_angle=poloidal_angle,
                        output_type= Output_Types[1])

                    path_output = input_path + filepath_format.format(
                        frequency=frequency, 
                        toroidal_angle=toroidal_angle, 
                        poloidal_angle=poloidal_angle,
                        output_type= Output_Types[2])

                    path_solver = input_path + filepath_format.format(
                        frequency=frequency, 
                        toroidal_angle=toroidal_angle, 
                        poloidal_angle=poloidal_angle,
                        output_type= Output_Types[3])

                    analysis_file = np.load(path_analysis)

                    path_len = len(analysis_file['distance_along_line'])

                    for key in ('distance_along_line', 'delta_theta_m', 'theta_m_output', 'theta_output', 'K_magnitude_array'):
                        current_ds = ds[key]
                        data = analysis_file[key]
                        if len(current_ds.dims) < 4:
                            ds[key] = current_ds.expand_dims(dim={'trajectory_step':np.arange(0, path_len)}, axis=-1).copy()
                        ds[key].loc[index] = data
                        print(ds[key])

                    ds['cutoff_index'].loc[index] = analysis_file['cutoff_index']
                    
                    analysis_file.close()

                    input_file = np.load(path_input)
                    
                    input_file.close()

                    output_file = np.load(path_output)

                    for key in ('q_R_array', 'q_zeta_array', 'q_Z_array', 
                                'K_R_array', 'K_Z_array', 'poloidal_flux_output'):
                        current_ds = ds[key]
                        if len(current_ds.dims) < 4:
                            ds[key] = current_ds.expand_dims(dim={'trajectory_step':np.arange(0, path_len)}, axis=-1).copy()
                        ds[key].loc[index] = output_file[key]
                        print(ds[key])

                    ds['K_zeta_initial'].loc[index] = output_file['K_zeta_initial']
                    output_file.close()
        
        self.dataset = ds
        self.spline_memo = {}
                
    # get methods

    def get_Dataset(self):
        return self.dataset

    def get_Dimensions(self):
        """Iterates over all variables in the Dataset and returns the shape
        of the corresponding DataArray in a dictionary
        """
        variables = self.get_Variables()
        variable_dict = {}
        for variable in variables:
            dims = self.dataset[variable].dims
            variable_dict[variable] = dims
            print(f'Dim({variable}) = {dims}')
        return variable_dict

    def get_Variables(self):
        return self.dataset.keys()



    # analysis sequences, output is saved in self.dataset but only runs
    # when method is called to avoid unnecessary computation
    # all methods are memoized
    
    def generate_probe_distance(self):
        data = self.dataset
        CUTOFF_ARRAY = data['cutoff_index']
        poloidal_distance = np.hypot(data['q_R_array'], data['q_Z_array'])
        self.dataset['poloidal_distance'] = poloidal_distance
        cutoff_distance =  poloidal_distance[CUTOFF_ARRAY]
        self.dataset['cutoff_distance'] = cutoff_distance
        return cutoff_distance

    def generate_probe_flux(self):
        return

    def generate_mismatch_at_cutoff(self):
        return

    def generate_opt_tor(self):
        """Uses numerical root finding on an interpolated spline to find
        the optimum toroidal steering with varying frequency and poloidal
        steering
        """
        return

    def generate_all(self):
        """Calls all analysis methods to generate and save analysis data
        """
        return

    # auxillary methods
    
    def create_spline(self, variable, dimension):
        """Memoized function that interpolates splines for
        any variable along arbitrary axes

        Args:
            variable (str): y-variable to fit the spline to
            dimension (str): dimension of the x-variable
        """
        args = (variable, dimension)
        if args in self.spline_memo.keys():
            return self.spline_memo[args]
        else:
            spline = None
            self.spline_memo[args] = spline
            return spline

    def gaussian(self, theta_m, delta):
        return np.exp(- ((theta_m/delta)**2))

    def noisy_gaussian(self, theta_m, delta, std = 0.05):
        mean = self.gaussian(theta_m, delta)
        return mean + np.random.normal(mean, std, len(theta_m))
