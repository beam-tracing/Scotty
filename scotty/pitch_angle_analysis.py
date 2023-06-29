"""Functions to load poloidal + toroidal + frequency sweeps of 
Scotty output data. Uses xarrays with poloidal angle, toroidal
angle and frequency as axes. 

"""

# Copyright 2023, Valerian Hall-Chen and Scotty contributors
# SPDX-License-Identifier: GPL-3.0


import xarray as xr
import numpy as np
from scipy.interpolate import (
    UnivariateSpline,
    RectBivariateSpline,
    RegularGridInterpolator,
)
from scipy.optimize import newton
import matplotlib.pyplot as plt


class SweepDataset:
    """Stores an xarray Dataset of Scotty output data with dimensions 'frequency',
    'poloidal_angle' and 'toroidal_angle'. Contains class methods for analysing
    and saving analysis data into the underlying Dataset.

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
                data_input, data_output, solver_output) in the filename.

    Output:
        None
    """

    def __init__(
        self,
        input_path,
        Frequencies,
        Poloidal_Angles,
        Toroidal_Angles,
        filepath_format,
        Output_Types=("analysis_output", "data_input", "data_output", "solver_output"),
    ):
        ds = xr.Dataset()
        variables = [
            "distance_along_line",
            "delta_theta_m",
            "theta_m_output",
            "theta_output",
            "cutoff_index",
            "q_R_array",
            "q_Z_array",
            "q_zeta_array",
            "K_magnitude_array",
            "K_R_array",
            "K_Z_array",
            "K_zeta_initial",
            "poloidal_flux_output",
        ]

        for variable in variables:
            ds[variable] = xr.DataArray(
                coords=[
                    ("frequency", Frequencies),
                    ("poloidal_angle", Poloidal_Angles),
                    ("toroidal_angle", Toroidal_Angles),
                ]
            )
        # print(ds)

        for frequency in Frequencies:
            for toroidal_angle in Toroidal_Angles:
                for poloidal_angle in Poloidal_Angles:
                    index = {
                        "frequency": frequency,
                        "poloidal_angle": poloidal_angle,
                        "toroidal_angle": toroidal_angle,
                    }

                    # File paths for each type of scotty output file
                    path_analysis = input_path + filepath_format.format(
                        frequency=frequency,
                        poloidal_angle=poloidal_angle,
                        toroidal_angle=toroidal_angle,
                        output_type=Output_Types[0],
                    )

                    path_input = input_path + filepath_format.format(
                        frequency=frequency,
                        poloidal_angle=poloidal_angle,
                        toroidal_angle=toroidal_angle,
                        output_type=Output_Types[1],
                    )

                    path_output = input_path + filepath_format.format(
                        frequency=frequency,
                        poloidal_angle=poloidal_angle,
                        toroidal_angle=toroidal_angle,
                        output_type=Output_Types[2],
                    )

                    path_solver = input_path + filepath_format.format(
                        frequency=frequency,
                        poloidal_angle=poloidal_angle,
                        toroidal_angle=toroidal_angle,
                        output_type=Output_Types[3],
                    )

                    # Load data into the empty Dataset

                    analysis_file = np.load(path_analysis)

                    path_len = len(analysis_file["distance_along_line"])

                    for key in (
                        "distance_along_line",
                        "delta_theta_m",
                        "theta_m_output",
                        "theta_output",
                        "K_magnitude_array",
                    ):
                        current_ds = ds[key]
                        data = analysis_file[key]
                        if len(current_ds.dims) < 4:
                            ds[key] = current_ds.expand_dims(
                                dim={"trajectory_step": np.arange(0, path_len)}, axis=-1
                            ).copy()
                        ds[key].loc[index] = data

                    ds["cutoff_index"].loc[index] = analysis_file["cutoff_index"]

                    analysis_file.close()

                    input_file = np.load(path_input)

                    input_file.close()

                    output_file = np.load(path_output)

                    for key in (
                        "q_R_array",
                        "q_zeta_array",
                        "q_Z_array",
                        "K_R_array",
                        "K_Z_array",
                        "poloidal_flux_output",
                    ):
                        current_ds = ds[key]
                        if len(current_ds.dims) < 4:
                            ds[key] = current_ds.expand_dims(
                                dim={"trajectory_step": np.arange(0, path_len)}, axis=-1
                            ).copy()
                        ds[key].loc[index] = output_file[key]

                    ds["K_zeta_initial"].loc[index] = output_file["K_zeta_initial"]
                    output_file.close()

        self.dataset = ds
        self.dataset["cutoff_index"] = ds["cutoff_index"].astype(int)
        self.dataset["theta_output"] = np.rad2deg(ds["theta_output"])
        self.dataset["theta_m_output"] = np.rad2deg(ds["theta_m_output"])
        self.dataset["delta_theta_m"] = np.rad2deg(ds["delta_theta_m"])
        self.spline_memo = {}

    #### Class Methods ####

    ## get methods

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
            print(f"Dim({variable}) = {dims}")
        return variable_dict

    def get_Variables(self):
        return list(self.dataset.keys())

    def print_KeysView(self):
        print(self.dataset.keys())

    def get_Coordinate_array(self, variable, dimension):
        data = self.dataset[variable]
        return data.coords[dimension].values

    ## Analysis sequences; output is saved as DataArrays in self.dataset but
    ## only runs when method is called to avoid unnecessary computation.
    ## All methods are memoized and thus can be called repeatedly to get the
    ## stored data.

    def generate_cutoff_distance(self):
        """Finds the poloidal cutoff distance of the beam from the
        magnetic axis. Saves a new 'poloidal_distance' 4-D DataArray for
        poloidal distance along trajectory in addition to cutoff distance.

        Returns:
            DataArray: A 3-D DataArray of cutoff distances with frequency,
            poloidal angle and toroidal angle as axes. Saves it with the
            key 'cutoff_distance'.
        """

        # TODO: fix this to calculate poloidal distance from magnetic axis
        data = self.dataset
        if "cutoff_distance" in self.get_Variables():
            print("cutoff_distance already generated.")
            return data["cutoff_index"]
        else:
            CUTOFF_ARRAY = data["cutoff_index"]
            # print(CUTOFF_ARRAY.shape)
            poloidal_distance = np.hypot(data["q_R_array"], data["q_Z_array"])
            self.dataset["poloidal_distance"] = poloidal_distance
            # print(poloidal_distance.shape)
            cutoff_distance = poloidal_distance.isel(trajectory_step=CUTOFF_ARRAY)
            self.dataset["cutoff_distance"] = cutoff_distance
            return cutoff_distance

    def generate_cutoff_flux(self):
        """Finds the poloidal cutoff flux of the beam. Poloidal flux analogue
        to generate_cutoff_distance.

        Returns:
            DataArray: A 3-D DataArray of cutoff distances with frequency,
            poloidal angle and toroidal angle as axes. Saves it with the
            key 'cutoff_flux'.
        """
        data = self.dataset
        if "cutoff_flux" in self.get_Variables():
            print("cutoff-flux already generated.")
            return data["cutoff_flux"]
        else:
            CUTOFF_ARRAY = data["cutoff_index"]
            cutoff_flux = data["poloidal_flux_output"].isel(
                trajectory_step=CUTOFF_ARRAY
            )
            self.dataset["cutoff_flux"] = cutoff_flux
            return cutoff_flux

    def generate_mismatch_at_cutoff(self):
        """Finds the mismatch angle ('theta_m') and mismatch tolerance
        ('delta_theta_m') at the cutoff point and saves it with the
        keys 'cutoff_theta_m' and 'cutoff_delta_theta_m'.

        Returns:
            Dataset: A Dataset of 2 DataArrays, one for 'cutoff_theta_m'
            and another for 'cutoff delta_theta_m', with frequency, poloidal
            angle and toroidal angle as axes.
        """
        data = self.dataset
        varlist = self.get_Variables()
        if ("cutoff_theta_m" in varlist) and ("cutoff_delta_theta_m" in varlist):
            print("mismatch_at_cutoff already generated.")
            return data[["cutoff_theta_m", "cutoff_delta_theta_m"]]
        else:
            CUTOFF_ARRAY = data["cutoff_index"]
            cutoff_theta_m = data["theta_m_output"].isel(trajectory_step=CUTOFF_ARRAY)
            self.dataset["cutoff_theta_m"] = cutoff_theta_m
            cutoff_delta_theta_m = data["delta_theta_m"].isel(
                trajectory_step=CUTOFF_ARRAY
            )
            self.dataset["cutoff_delta_theta_m"] = cutoff_delta_theta_m
            return self.dataset[["cutoff_theta_m", "cutoff_delta_theta_m"]]

    def generate_opt_tor(self):
        """Uses scipy.optimize.newton on an interpolated spline to find
        the optimum toroidal steering with varying frequency and poloidal
        steering and saves it with the key 'opt_tor'.

        Returns:
            DataArray: A 2-D DataArray of optimal toroidal steering angles (in degrees)
        """
        if "opt_tor" in self.get_Variables():
            print("opt-tor already generated.")
            return self.dataset["opt_tor"]
        else:
            Frequencies = self.get_Coordinate_array("cutoff_theta_m", "frequency")
            Poloidal_Angles = self.get_Coordinate_array(
                "cutoff_theta_m", "poloidal_angle"
            )
            opt_tor_array = xr.DataArray(
                coords=[
                    ("frequency", Frequencies),
                    ("poloidal_angle", Poloidal_Angles),
                ]
            )
            for frequency in Frequencies:
                for poloidal_angle in Poloidal_Angles:
                    coords_dict = {
                        "frequency": frequency,
                        "poloidal_angle": poloidal_angle,
                    }
                    spline = self.create_1Dspline(
                        "cutoff_theta_m", "toroidal_angle", coords_dict
                    )
                    root = newton(spline, x0=0, maxiter=50)
                    opt_tor_array.loc[coords_dict] = root
            self.dataset["opt_tor"] = opt_tor_array
            return opt_tor_array

    def generate_reflection_mask(self):
        """Function that checks if the ray is reflected, and stores the result
        in a 3-D DataArray mask with standard axes. The value False signifies reflection
        (angle between launch and exit directions is obtuse) and True signifies no
        reflection (angle between launch and exit directions is acute). To be used
        as with numpy MaskedArray, where True booleans signify invalid data.

        Returns:
            DataArray: A DataArray of booleans, saved in the Dataset with the
            key 'reflection_mask'.
        """
        data = self.dataset
        if "reflection_mask" in self.get_Variables():
            print("reflection_mask already generated")
            return self.dataset["reflection_mask"]
        else:
            Frequencies = self.get_Coordinate_array("distance_along_line", "frequency")
            Poloidal_Angles = self.get_Coordinate_array(
                "distance_along_line", "poloidal_angle"
            )
            Toroidal_Angles = self.get_Coordinate_array(
                "distance_along_line", "toroidal_angle"
            )
            reflection_mask = xr.DataArray(
                coords=[
                    ("frequency", Frequencies),
                    ("poloidal_angle", Poloidal_Angles),
                    ("toroidal_angle", Toroidal_Angles),
                ]
            )

            # TODO: Look up if this can be done with array methods instead
            for frequency in Frequencies:
                for poloidal_angle in Poloidal_Angles:
                    for toroidal_angle in Toroidal_Angles:
                        coords_dict = {
                            "frequency": frequency,
                            "poloidal_angle": poloidal_angle,
                            "toroidal_angle": toroidal_angle,
                        }
                        q_R = data["q_R_array"].loc[coords_dict]
                        q_Z = data["q_Z_array"].loc[coords_dict]
                        q_zeta = data["q_zeta_array"].loc[coords_dict]

                        # Calculate initial launch vector
                        del_qR0 = q_R[1] - q_R[0]
                        del_qZ0 = q_Z[1] - q_Z[0]
                        del_qzeta0 = q_zeta[1] - q_zeta[0]
                        launch_vector = np.array((del_qR0, del_qZ0, del_qzeta0))

                        # Calculate final exit vector
                        del_qR1 = q_R[-1] - q_R[-2]
                        del_qZ1 = q_Z[-1] - q_Z[-2]
                        del_qzeta1 = q_zeta[-1] - q_zeta[-2]
                        exit_vector = np.array((del_qR1, del_qZ1, del_qzeta1))

                        inner_product = np.inner(launch_vector, exit_vector)

                        if inner_product > 0:  # Angle is acute (straight through)
                            mask_value = True
                        else:  # Angle is obtuse or orthogonal (reflection)
                            mask_value = False
                        reflection_mask.loc[coords_dict] = mask_value

            self.dataset["reflection_mask"] = reflection_mask
            return reflection_mask

    def generate_all(self):
        """Calls all analysis methods to generate and save analysis data.
        Returns a list of variables including newly-generated ones that
        can be indexed.
        """
        self.generate_cutoff_distance()
        self.generate_cutoff_flux()
        self.generate_mismatch_at_cutoff()
        self.generate_opt_tor()
        self.generate_reflection_mask()
        return self.get_Variables()

    ## simulation methods

    def get_simulated_power_prof(
        self, frequency, poloidal_angle, noise_flag=True, std=0.05
    ):
        """Returns a profile function that describes the simulated backscattered power fraction
        as a function of toroidal steering. Defaults to random gaussian noise.

        Args:
            frequency (int/float): Frequency of toroidal scan being simulated.
            poloidal_angle (int/float): Poloidal launch angle of toroidal scan being simulated.
            noise_flag (bool, optional): Handles whether a smooth or noisy profile is returned. Defaults to True.
            std (float, optional): Normalised standard deviation for gaussian noise. Defaults to 0.05.

        Returns:
            Callable[[float], float]: Power profile function of toroidal angle
        """
        coords_dict = {
            "frequency": frequency,
            "poloidal_angle": poloidal_angle,
        }
        theta_m_spline = self.create_1Dspline(
            "cutoff_theta_m", "toroidal_angle", coords_dict
        )
        delta_spline = self.create_1Dspline(
            "cutoff_delta_theta_m", "toroidal_angle", coords_dict
        )

        def simulated_power_prof(toroidal_angle):
            delta = delta_spline(toroidal_angle)
            theta_m = theta_m_spline(toroidal_angle)
            if noise_flag:
                return self.noisy_gaussian(theta_m, delta)
            else:
                return self.gaussian(theta_m, delta)

        return simulated_power_prof

    ## auxillary methods

    def create_1Dspline(self, variable, dimension, coords_dict):
        """Memoized function that interpolates splines for any variable along
        a single arbitrary axis with scipy.interpolate.UnivariateSpline.

        Args:
            variable (str): index the specific DataArray to fit the spline to
            dimension (str): dimension name of the x-coordinate
            coords_dict (dict): a dictionary of the form {dimension: value} for all
            other dimensions apart from the x-coordinate dimension. For example, if
            the spline is for interpolating theta_m as a function of toroidal_angle,
            coordinate values for all other dimensions of theta_m (frequency and
            poloidal angle) must be supplied.

        Returns:
            Callable: An interpolated UnivariateSpline function.
        """
        coords_hashable = self._dict_to_hashable(coords_dict)
        args = (variable, dimension, coords_hashable)
        if args in self.spline_memo.keys():
            return self.spline_memo[args]
        else:
            data = self.dataset[variable]
            x_coordinates = self.get_Coordinate_array(variable, dimension)
            y_data = data.loc[coords_dict]
            y_values = y_data.values
            spline = UnivariateSpline(x=x_coordinates, y=y_values, s=0)
            self.spline_memo[args] = spline
            return spline

    def create_2Dspline(self, variable, xdimension, ydimension, coords_dict):
        """Memoized function that interpolates splines for any variable along
        two arbitrary axes with scipy.interpolate.RectBivariateSpline.

        Args:
            variable (str): index the specific DataArray to fit the spline to
            xdimension (str): dimension name of the x-coordinate
            ydimension (str): dimension name of the y-coordinate
            coords_dict (dict): a dictionary of the form {dimension: value} for all
            other dimensions apart from the x and y coordinate dimensions. See
            create_1Dspline for a similar method.

        Returns:
            Callable: An interpolated RectBivariateSpline function.
        """
        coords_hashable = self._dict_to_hashable(coords_dict)
        args = (variable, xdimension, ydimension, coords_hashable)
        if args in self.spline_memo.keys():
            return self.spline_memo[args]
        else:
            data = self.dataset[variable]
            x_coordinates = self.get_Coordinate_array(variable, xdimension)
            y_coordinates = self.get_Coordinate_array(variable, ydimension)
            z_values = data.loc[coords_dict].transpose(xdimension, ydimension).values
            spline = RectBivariateSpline(
                x=x_coordinates, y=y_coordinates, z=z_values, s=0
            )
            self.spline_memo[args] = spline
            return spline

    def gaussian(self, theta_m, delta):
        return np.exp(-((theta_m / delta) ** 2))

    def noisy_gaussian(self, theta_m, delta, std=0.05):
        mean = self.gaussian(theta_m, delta)
        return mean + np.random.normal(mean, std, len(theta_m))

    # To be implemented
    """
    def fit_gaussian(self, simulated_power_prof):
        return
    """

    def _dict_to_hashable(self, dictionary):
        temp_list = []
        for key in dictionary.keys():
            value = dictionary[key]
            temp_list.append((key, value))
        hashable = tuple(temp_list)
        return hashable

    def _hashable_to_dict(self, hashable):
        dictionary = {}
        for pair in hashable:
            key = pair[0]
            value = pair[1]
            dictionary[key] = value
        return dictionary

    ## basic plotting methods

    def plot_cutoff_contour(
        self,
        const_angle_str,
        const_angle,
        save_path=None,
        measure="rho",
        mask_flag=True,
        bounds=None,
    ):
        """Creates a contour plot of toroidal/poloidal steering vs. frequency, with
        contours of constant cutoff (either rho or metres). A bivariate spline is fitted
        to the data to produce the plot.

        Args:
            const_angle_str (str): Specify if frequency is being plotted against
            'toroidal_angle' or 'poloidal_angle'.
            const_angle (int/float): Value of poloidal/toroidal angle to be held
            constant.
            save_path (str, optional): If provided, saves the figure to the file
            path specified. Default is None.
            measure (str, optional): Specify 'rho' or 'm' to plot in terms of normalized
            poloidal radius or actual poloidal distance. Default is 'rho'.
            mask_flag (bool, optional): Select whether to mask the data with 'reflection_mask'.
            Default is True.
            bounds (tup, optional): Specify plotting range in the form (x_start, x_stop,
            y_start, y_stop)

        Returns:
            Tuple: A tuple of the X, Y, Z and mask arrays used to plot the contours
        """
        if const_angle_str == "toroidal_angle":
            var_angle_str = "poloidal_angle"
        elif const_angle_str == "poloidal_angle":
            var_angle_str = "toroidal_angle"
        else:
            raise ValueError(
                "var_angle_str must be 'toroidal_angle' or 'poloidal_angle'."
            )

        if measure == "rho":
            title_label = "rho"
            variable = "cutoff_flux"
        elif measure == "m":
            title_label = "dist"
            variable = "cutoff_distance"
        else:
            raise ValueError("Provided measure is not 'rho' or 'm'.")

        if const_angle not in self.get_Coordinate_array(variable, const_angle_str):
            raise ValueError("Value of const_angle not in original coordinate array.")

        coords_dict = {const_angle_str: const_angle}
        xdimension = "frequency"
        ydimension = var_angle_str
        spline_surface = self.create_2Dspline(
            variable, xdimension, ydimension, coords_dict
        )

        x_coordinates = self.get_Coordinate_array(variable, xdimension)
        y_coordinates = self.get_Coordinate_array(variable, ydimension)

        if bounds == None:
            x_start = x_coordinates[0]
            x_stop = x_coordinates[-1]
            y_start = y_coordinates[0]
            y_stop = y_coordinates[-1]
        else:
            x_start = bounds[0]
            x_stop = bounds[1]
            y_start = bounds[2]
            y_stop = bounds[3]

        x_array = np.linspace(x_start, x_stop, num=30)
        y_array = np.linspace(y_start, y_stop, num=30)
        Xgrid, Ygrid = np.meshgrid(x_array, y_array, indexing="ij")
        Zgrid = spline_surface(x_array, y_array)

        # Convert from poloidal flux to rho
        if measure == "rho":
            Zgrid = np.sqrt(Zgrid)

        # Interpolate and apply the reflection mask to remove inaccessible regions
        reflection_mask = (
            self.dataset["reflection_mask"]
            .loc[coords_dict]
            .transpose(xdimension, ydimension)
            .values
        )
        mask_function = RegularGridInterpolator(
            (x_coordinates, y_coordinates),
            reflection_mask,
            method="nearest",
            bounds_error=False,
        )
        interp_mask = mask_function((Xgrid, Ygrid))
        Z_masked = np.ma.array(Zgrid, mask=interp_mask)

        fig, ax = plt.subplots()
        if mask_flag:
            CS = ax.contour(
                Xgrid, Ygrid, Z_masked, levels=20, cmap="plasma_r", corner_mask=True
            )
        else:
            CS = ax.contour(
                Xgrid, Ygrid, Zgrid, levels=20, cmap="plasma_r", corner_mask=True
            )

        fig.suptitle(f"Cutoff {title_label}, {const_angle_str}={const_angle}$^\circ$")
        ax.set_xlabel("frequency/GHz")
        ax.set_ylabel(f"{var_angle_str}/$^\circ$")
        ax.clabel(CS, inline=True, fontsize=6)
        fig.colorbar(CS)
        if save_path:
            plt.savefig(
                save_path + f"cutoff_contour_{const_angle_str}{const_angle}.jpg",
                dpi=200,
            )
        return (Xgrid, Ygrid, Zgrid, interp_mask)

    # To be implemented
    """ 
    def plot_cutoff_angles(self, poloidal_angle):
        return

    def plot_opt_tor_vs_freq(self, poloidal_angle):
        return
    """
