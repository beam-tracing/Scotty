"""Functions to load poloidal + toroidal + frequency sweeps of 
Scotty output data. Uses xarrays with poloidal angle, toroidal
angle and frequency as axes. 

"""

# Copyright 2023, Valerian Hall-Chen and Scotty contributors
# SPDX-License-Identifier: GPL-3.0


import xarray as xr
import numpy as np
import json
from scipy.interpolate import (
    UnivariateSpline,
    RectBivariateSpline,
    RegularGridInterpolator,
)
from scipy.optimize import newton
from scipy.stats import norm
import matplotlib.pyplot as plt
from scotty.fun_general import find_q_lab_Cartesian
from itertools import product


class SweepDataset:
    """Stores an xarray Dataset of Scotty output data with dimensions 'frequency',
    'poloidal_angle' and 'toroidal_angle'. Contains class methods for analysing
    and saving analysis data into the underlying Dataset. Analysis methods only
    run when method is called to avoid unnecessary computation. All methods are
    memoized and thus can be called repeatedly to get the stored data.

    Class is constructed by calling classmethods from_netcdf or from_scotty.
    """

    def __init__(self, ds):
        self.dataset = ds
        self.spline_memo = {}

    # I/O

    @classmethod
    def from_netcdf(cls, path_string):
        """Constructs a SweepDataset instance from an exported netCDF file.

        Args:
            path_string (str): Path to .nc file

        """
        with xr.open_dataset(path_string) as ds:
            class_type = ds.attrs["class_type"]
            if class_type != "SweepDataset":
                raise ValueError(
                    ".nc file does not have class_type = 'SweepDataset' attribute"
                )
            classobject = cls(ds)
            classobject.missing_indices = json.loads(ds.attrs["missing_indices"])
            # list(map(json.loads, ds.attrs["missing_indices"]))
        return classobject

    @classmethod
    def from_scotty(
        cls,
        input_path,
        frequencies,
        poloidal_angles,
        toroidal_angles,
        filepath_format,
        output_types=("analysis_output", "data_input", "data_output", "solver_output"),
        attrs_dict={},
    ):
        """Constructs a SweepDataset instance from a set of Scotty output files.

        Args:
            input_path (string): Path to home folder containing all outputs
            frequencies (iterable): Array or List of indexing frequencies
            toroidal_angles (iterable): Array or List of indexing toroidal launch angles
            poloidal_angles (iterable): Array or List of indexing poloidal launch angles
            filepath_format (f-string): an f-string that specifies the path string to be
            appended to the end of the input_path to access outputs of individual runs.
            Note that the file path or filenames must include {frequency}, {toroidal_angle},
            and {poloidal_angle} specified in frequencies, toroidal_angles, and poloidal_angles,
            as well as a {output_types} corresponding to the type of scotty output (analysis_output,
            data_input, data_output, solver_output) in the filename.
        """
        ds = xr.Dataset()
        ds.attrs["class_type"] = "SweepDataset"
        ds.attrs["descriptor"] = ""
        for key, value in attrs_dict:
            ds.attrs[key] = value

        missing_indices = []

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

        input_attributes = [
            "launch_position",
            "launch_curvature",
            "mode_flag",
            "data_R_coord",
            "data_Z_coord",
            "poloidalFlux_grid",
            "ne_data_density_array",
            "ne_data_radialcoord_array",
        ]

        print("Reading .npz files...")

        for variable in variables:
            ds[variable] = xr.DataArray(
                coords=[
                    ("frequency", frequencies),
                    ("poloidal_angle", poloidal_angles),
                    ("toroidal_angle", toroidal_angles),
                ]
            )

        for combination in product(frequencies, toroidal_angles, poloidal_angles):
            frequency, toroidal_angle, product = combination
            index = {
                "frequency": float(frequency),
                "poloidal_angle": float(poloidal_angle),
                "toroidal_angle": float(toroidal_angle),
            }

            # File paths for each type of scotty output file
            path_analysis = input_path + filepath_format.format(
                frequency=frequency,
                poloidal_angle=poloidal_angle,
                toroidal_angle=toroidal_angle,
                output_type=output_types[0],
            )

            path_input = input_path + filepath_format.format(
                frequency=frequency,
                poloidal_angle=poloidal_angle,
                toroidal_angle=toroidal_angle,
                output_type=output_types[1],
            )

            path_output = input_path + filepath_format.format(
                frequency=frequency,
                poloidal_angle=poloidal_angle,
                toroidal_angle=toroidal_angle,
                output_type=output_types[2],
            )

            path_solver = input_path + filepath_format.format(
                frequency=frequency,
                poloidal_angle=poloidal_angle,
                toroidal_angle=toroidal_angle,
                output_type=output_types[3],
            )

            # Load data into the empty Dataset

            try:
                with np.load(path_analysis) as analysis_file:
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

                # Currently unutiilized
                """
                with np.load(path_input) as input_file:
                    pass
                """

                with np.load(path_output) as output_file:
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

            except FileNotFoundError:
                print(
                    f"No file found for freq={frequency} pol={poloidal_angle}, tor={toroidal_angle}"
                )
                missing_indices.append(index)
                continue

            # Load flux geometry information from the first available scotty file
            for combination in product(frequencies, toroidal_angles, poloidal_angles):
                frequency, toroidal_angle, product = combination
                index = {
                    "frequency": float(frequency),
                    "poloidal_angle": float(poloidal_angle),
                    "toroidal_angle": float(toroidal_angle),
                }

                path_analysis = input_path + filepath_format.format(
                    frequency=frequency,
                    poloidal_angle=poloidal_angle,
                    toroidal_angle=toroidal_angle,
                    output_type=output_types[0],
                )

                path_input = input_path + filepath_format.format(
                    frequency=frequency,
                    poloidal_angle=poloidal_angle,
                    toroidal_angle=toroidal_angle,
                    output_type=output_types[1],
                )

                try:
                    with np.load(path_input) as input_file:
                        for attribute in input_attributes:
                            ds.attrs[attribute] = input_file[attribute]

                    with np.load(path_analysis) as analysis_file:
                        ds.attrs["poloidal_flux_on_midplane"] = analysis_file[
                            "poloidal_flux_on_midplane"
                        ]
                        ds.attrs["R_midplane_points"] = analysis_file[
                            "R_midplane_points"
                        ]
                        index = ds.attrs["poloidal_flux_on_midplane"].argmin()
                        R_coord = ds.attrs["R_midplane_points"][index]
                        ds.attrs["magnetic_axis_RZ"] = np.array((R_coord, 0.0))

                    print(f"Input information read from {path_input}")
                    break

                except FileNotFoundError:
                    continue

        print("File reading complete.")

        ds["cutoff_index"] = ds["cutoff_index"].astype(int)
        ds["theta_output"] = np.rad2deg(ds["theta_output"])
        ds["theta_m_output"] = np.rad2deg(ds["theta_m_output"])
        ds["delta_theta_m"] = np.rad2deg(ds["delta_theta_m"])

        classobject = cls(ds)
        classobject.missing_indices = missing_indices

        # Interpolate missing data where solver couldn't reach completion
        print("Interpolating missing values...")
        classobject.check_cutoff_indices()
        classobject.check_all_float_arrays()
        print("Interpolation complete. Check output for unsuccessful interpolations.")

        return classobject

    def to_netcdf(self, folder="", filename=None, suffix=""):
        """Saves contents of Dataset into a netCDF4 file for easy read and writing.

        Args:
            folder (str, optional): Path to folder to save file to. Default is None.
            filename (str, optional): Filename to save the nc file as. Default is 'SweepDataset', or
            the descriptor string if provided.
        """
        self.dataset.attrs["missing_indices"] = json.dumps(self.missing_indices)
        '''
        if not filename:
            descriptor = self.dataset.attrs["descriptor"]
            if descriptor:
                filename = descriptor
            else:
                filename = "SweepDataset"'''

        filename = filename or self.dataset.get("descriptor", "SweepDataset")

        file_path = f"{folder}{filename}{suffix}.nc"
        self.dataset.to_netcdf(path=file_path)

    #### Set Methods ####

    def set_attrs(self, attribute, value):
        self.dataset.attrs[attribute] = value

    def set_descriptor(self, descriptor):
        """Used to set a unique descriptor to identify sweeps with different plasma
        equilibrium.
        """
        self.set_attrs("descriptor", descriptor)

    ## get methods/properties

    def get_Dataset_copy(self):
        """
        Returns:
            Dataset: Deep copy of the underlying xarray Dataset
        """
        return self.dataset.copy(deep=True)

    def get_coordinate_array(self, dimension):
        data = self.dataset
        return data.coords[dimension].values

    @property
    def dimensions(self):
        """A dictionary of Dataset variables and their corresponding dimensions."""
        variables = self.variables
        variable_dict = {}
        for variable in variables:
            dims = self.dataset[variable].dims
            variable_dict[variable] = dims
            print(f"Dim({variable}) = {dims}")
        return variable_dict

    @property
    def variables(self):
        return list(self.dataset.keys())

    @property
    def descriptor(self):
        return str(self.dataset.attrs["descriptor"])

    def print_KeysView(self):
        print(self.dataset.keys())

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
        if "cutoff_distance" in self.variables:
            print("cutoff_distance already generated.")
            return data["cutoff_index"]

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
        if "cutoff_flux" in self.variables:
            print("cutoff-flux already generated.")
            return data["cutoff_flux"]

        CUTOFF_ARRAY = data["cutoff_index"]
        cutoff_flux = data["poloidal_flux_output"].isel(trajectory_step=CUTOFF_ARRAY)
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
        varlist = self.variables
        if ("cutoff_theta_m" in varlist) and ("cutoff_delta_theta_m" in varlist):
            print("mismatch_at_cutoff already generated.")
            return data[["cutoff_theta_m", "cutoff_delta_theta_m"]]

        CUTOFF_ARRAY = data["cutoff_index"]
        cutoff_theta_m = data["theta_m_output"].isel(trajectory_step=CUTOFF_ARRAY)
        self.dataset["cutoff_theta_m"] = cutoff_theta_m
        cutoff_delta_theta_m = data["delta_theta_m"].isel(trajectory_step=CUTOFF_ARRAY)
        self.dataset["cutoff_delta_theta_m"] = cutoff_delta_theta_m
        return self.dataset[["cutoff_theta_m", "cutoff_delta_theta_m"]]

    def generate_opt_tor(self, mask_flag=True):
        """Uses scipy.optimize.newton on an interpolated spline to find
        the optimum toroidal steering with varying frequency and poloidal
        steering and saves it with the key 'opt_tor'.

        Args:
            mask_flag (bool): Default True. Determines whether the optimum
            toroidal angle calculated will be reflected or not and saves the
            result as a boolean mask DataArray 'opt_tor_mask'.

        Returns:
            DataArray: A 2-D DataArray of optimal toroidal steering angles (in degrees)
        """
        if "opt_tor" in self.variables:
            print("opt-tor already generated.")
            return self.dataset["opt_tor"]

        if "cutoff_theta_m" not in self.variables:
            print(
                "Pre-req variable 'mismatch_at_cutoff' is not yet generated. Running generate_mismatch_at_cutoff..."
            )
            self.generate_mismatch_at_cutoff()

        frequencies = self.get_coordinate_array("frequency")
        poloidal_angles = self.get_coordinate_array("poloidal_angle")
        toroidal_angles = self.get_coordinate_array("toroidal_angle")
        opt_tor_array = xr.DataArray(
            coords=[
                ("frequency", frequencies),
                ("poloidal_angle", poloidal_angles),
            ]
        )

        for frequency in frequencies:
            for poloidal_angle in poloidal_angles:
                coords_dict = {
                    "frequency": frequency,
                    "poloidal_angle": poloidal_angle,
                }
                spline = self.create_1Dspline(
                    "cutoff_theta_m", "toroidal_angle", coords_dict
                )
                try:
                    root = newton(spline, x0=0, fprime=spline.derivative(), maxiter=100)
                    opt_tor_array.loc[coords_dict] = root
                except RuntimeError as error:
                    print(
                        f"No zero found for Freq={frequency} GHz, Pol={poloidal_angle} deg: ",
                        error,
                    )
                    continue

        no_opt_tor_mask = opt_tor_array.isnull()
        self.dataset["opt_tor"] = opt_tor_array

        if mask_flag:

            def vfunc(arg):
                func = lambda x: find_nearest(toroidal_angles, x)
                return xr.apply_ufunc(func, arg, vectorize=True)

            closest_tor_array = vfunc(opt_tor_array)
            reflection_mask = self.generate_reflection_mask().transpose(
                "frequency", "poloidal_angle", "toroidal_angle"
            )
            reflected_opt_tor_mask = reflection_mask.isel(
                toroidal_angle=closest_tor_array
            )

        self.dataset["opt_tor_mask"] = np.logical_and(
            no_opt_tor_mask, reflected_opt_tor_mask
        )

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
        if "reflection_mask" in self.variables:
            print("reflection_mask already generated")
            return self.dataset["reflection_mask"]

        frequencies = self.get_coordinate_array("frequency")
        poloidal_angles = self.get_coordinate_array("poloidal_angle")
        toroidal_angles = self.get_coordinate_array("toroidal_angle")

        reflection_mask = xr.DataArray(
            coords=[
                ("frequency", frequencies),
                ("poloidal_angle", poloidal_angles),
                ("toroidal_angle", toroidal_angles),
            ]
        ).astype(bool)

        q_R = data["q_R_array"]
        q_Z = data["q_Z_array"]
        q_zeta = data["q_zeta_array"]

        # Calculate initial launch vector

        del_qR0 = q_R.isel({"trajectory_step": 1}) - q_R.isel({"trajectory_step": 0})
        del_qZ0 = q_Z.isel({"trajectory_step": 1}) - q_Z.isel({"trajectory_step": 0})
        del_qzeta0 = q_zeta.isel({"trajectory_step": 1}) - q_zeta.isel(
            {"trajectory_step": 0}
        )
        launch_vector = np.array((del_qR0, del_qZ0, del_qzeta0))
        launch_cart = find_q_lab_Cartesian(launch_vector)

        # Calculate final exit vector
        del_qR1 = q_R.isel({"trajectory_step": -1}) - q_R.isel({"trajectory_step": -2})
        del_qZ1 = q_Z.isel({"trajectory_step": -1}) - q_Z.isel({"trajectory_step": -2})
        del_qzeta1 = q_zeta.isel({"trajectory_step": -1}) - q_zeta.isel(
            {"trajectory_step": -2}
        )
        exit_vector = np.array((del_qR1, del_qZ1, del_qzeta1))
        exit_cart = find_q_lab_Cartesian(exit_vector)

        dimensions = del_qR0.dims

        inner_product_array = (
            np.multiply(launch_cart[0], exit_cart[0])
            + np.multiply(launch_cart[1], exit_cart[1])
            + np.multiply(launch_cart[2], exit_cart[2])
        )
        # np.inner(launch_cart, exit_cart)

        # True if angle is acute (straight through), false if obtuse
        # or orthogonal (reflection)
        reflection_mask = inner_product_array > 0
        self.dataset["reflection_mask"] = (dimensions, reflection_mask)
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
        return self.variables

    ## simulation methods

    def get_simulated_power_prof(
        self,
        frequency,
        poloidal_angle,
        std_noise=0.05,
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

        theta_m_spline = self.create_3Dspline(
            variable="cutoff_theta_m",
            xdimension="frequency",
            ydimension="poloidal_angle",
            zdimension="toroidal_angle",
        )
        delta_spline = self.create_3Dspline(
            variable="cutoff_delta_theta_m",
            xdimension="frequency",
            ydimension="poloidal_angle",
            zdimension="toroidal_angle",
        )

        def simulated_power_prof(toroidal_angle, std=std_noise, noise_flag=True):
            delta = delta_spline((frequency, poloidal_angle, toroidal_angle))
            theta_m = theta_m_spline((frequency, poloidal_angle, toroidal_angle))
            if noise_flag:
                return noisy_gaussian(theta_m, delta, std=std)
            else:
                return gaussian(theta_m, delta)

        return simulated_power_prof

    ## auxillary methods

    def create_1Dspline(self, variable, dimension, coords_dict={}):
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
            x_coordinates = self.get_coordinate_array(dimension)
            y_data = data.loc[coords_dict]
            y_values = y_data.values
            spline = UnivariateSpline(x=x_coordinates, y=y_values, s=0)
            self.spline_memo[args] = spline
            return spline

    def create_2Dspline(self, variable, xdimension, ydimension, coords_dict={}):
        """Memoized function that interpolates splines for any variable along
        two arbitrary axes with linear interpolation (scipy RegularGridInterpolator).

        Args:
            variable (str): index the specific DataArray to fit the spline to
            xdimension (str): dimension name of the x-coordinate
            ydimension (str): dimension name of the y-coordinate
            coords_dict (dict): a dictionary of the form {dimension: value} for all
            other dimensions apart from the x and y coordinate dimensions. See
            create_1Dspline for a similar method.

        Returns:
            Callable: A RegularGridInterpolator function.
        """
        coords_hashable = self._dict_to_hashable(coords_dict)
        args = (variable, xdimension, ydimension, coords_hashable)
        if args in self.spline_memo.keys():
            return self.spline_memo[args]
        else:
            data = self.dataset[variable]
            x_coordinates = self.get_coordinate_array(xdimension)
            y_coordinates = self.get_coordinate_array(ydimension)
            z_values = data.loc[coords_dict].transpose(xdimension, ydimension).values
            spline = RegularGridInterpolator(
                (x_coordinates, y_coordinates), z_values, method="linear", #TODO: Change back to pchip once done
            )
            self.spline_memo[args] = spline
            return spline

    def create_3Dspline(
        self, variable, xdimension, ydimension, zdimension, coords_dict={}
    ):
        """Memoized function that interpolates splines for any variable along
        two arbitrary axes with scipy.interpolate.RegularGridInterpolator.
        Interpolation method is linear.

        Args:
            variable (str): index the specific DataArray to fit the spline to
            xdimension (str): dimension name of the x-coordinate
            ydimension (str): dimension name of the y-coordinate
            zdimensoin (str): dimension name of the z-coordinate
            coords_dict (dict): a dictionary of the form {dimension: value} for all
            other dimensions apart from the x and y coordinate dimensions. See
            create_1Dspline for a similar method.

        Returns:
            Callable: An RegularGridInterpolator function.
        """
        coords_hashable = self._dict_to_hashable(coords_dict)
        args = (variable, xdimension, ydimension, coords_hashable)
        if args in self.spline_memo.keys():
            return self.spline_memo[args]
        else:
            data = self.dataset[variable]
            x_coordinates = self.get_coordinate_array(xdimension)
            y_coordinates = self.get_coordinate_array(ydimension)
            z_coordinates = self.get_coordinate_array(zdimension)
            func_values = (
                data.loc[coords_dict]
                .transpose(xdimension, ydimension, zdimension)
                .values
            )
            spline = RegularGridInterpolator(
                points=(x_coordinates, y_coordinates, z_coordinates),
                values=func_values,
                method="linear",
            )
            self.spline_memo[args] = spline
            return spline

    def check_reflected(self, frequency, toroidal_angle, poloidal_angle):
        """Checks whether a ray of an arbitrary frequency, toroidal and poloidal steering
        will be reflected from the plasma based on nearest grid points.
        """

        frequencies = self.get_coordinate_array("frequency")
        poloidal_angles = self.get_coordinate_array("poloidal_angle")
        toroidal_angles = self.get_coordinate_array("toroidal_angle")

        # Find nearest valid coordinate points
        freq_coord = find_nearest(frequencies, frequency)
        tor_coord = find_nearest(toroidal_angles, toroidal_angle)
        pol_coord = find_nearest(poloidal_angles, poloidal_angle)

        coords_dict = {
            "frequency": freq_coord,
            "toroidal_angle": tor_coord,
            "poloidal_angle": pol_coord,
        }

        reflection_mask = self.generate_reflection_mask()
        return reflection_mask[coords_dict]

    def check_cutoff_indices(self):
        """Checks for and interpolates problematic cutoff indices (negative integers that
        occur when casting np.NaN to int).

        Does not interpolate if gap is larger than 3 data points to account for situations
        where the simulation parameters go out of bounds.
        """
        cutoff_indices = self.dataset["cutoff_index"]
        # cast to float to work with np.NaN values
        float_indices = cutoff_indices.astype(float)
        index_list = np.argwhere(float_indices.values < 0)
        print(f"Number of problematic indices: {len(index_list)}")
        interp_indices = float_indices.where(float_indices >= 0)
        for dimension in ("toroidal_angle", "poloidal_angle", "frequency"):
            try:
                interp_indices = interp_indices.interpolate_na(
                    dim=dimension, method="nearest", max_gap=3
                )
            except Exception:
                print(
                    f"Indice interpolation along {dimension} failed. Trying new axis..."
                )
                continue
        interp_indices = interp_indices.astype(int)
        new_list = np.argwhere(interp_indices.values < 0)
        self.dataset["cutoff_index"] = interp_indices
        if len(new_list):
            print(
                f"Failed to resolve problematic indices. Consider masking the affected regions. Failed indices: {new_list}"
            )
            return new_list

    def check_float_arrays(self, variable):
        """Checks for and interpolates any null array values.

        Does not interpolate if gap is larger than 3 data points to account for situations
        where the simulation parameters go out of bounds.
        """
        new_array = self.dataset[variable]
        print("Number of NaN entries:", len(np.argwhere(new_array.isnull().values)))
        # Try multiple dimensions as interpolating in one axis may not eliminate all gaps
        for dimension in ("toroidal_angle", "poloidal_angle", "frequency"):
            try:
                new_array = new_array.interpolate_na(
                    dim=dimension, method="cubic", max_gap=3
                )
            except Exception:
                print(
                    f"{variable} interpolation along {dimension} failed. Trying new axis..."
                )
                continue

        index_list = np.argwhere(new_array.isnull().values)
        self.dataset[variable] = new_array

        if len(index_list):
            print(
                f"Failed to resolve problematic {variable} values. Consider masking the affected regions. Failed indices: {index_list}"
            )
            return index_list

    def check_all_float_arrays(self):
        for variable in self.variables:
            print(f"Checking {variable}...")
            if variable != "cutoff_index":
                self.check_float_arrays(variable)

    def _dict_to_hashable(self, dictionary):
        return tuple(dictionary.items())

    def _hashable_to_dict(self, hashable):
        return dict(hashable)

    ## basic plotting methods

    def imshow_slice(
        self,
        variable,
        xdimension,
        ydimension,
        coords_dict={},
        cmap="plasma_r",
        **kwargs,
    ):
        """Visualizes a specified slice of the DataArray with plt.imshow.

        Args:
            variable(str): DataArray variable to visualize
            xdimension (str): Dimension for the x-axis
            ydimension (str): Dimension for the y-axis
            coords_dict (dict): Dictionary of coordinate values of the other
            dimensions to be held constant
            cmap (str): Colormap to be used
            kwargs (optional): Optional keyword arguments to pass to plt.imshow
        """
        data_slice = self.dataset[variable].loc[coords_dict]
        if len(data_slice.shape) != 2:
            raise ValueError("Inappropriate number of constant dimensions supplied")
        title_string = ""
        for key, value in coords_dict.items():
            title_string += f",{key}={value} "
        x_coords = self.get_coordinate_array(xdimension)
        y_coords = self.get_coordinate_array(ydimension)
        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
        data_slice = data_slice.transpose(ydimension, xdimension)
        fig, ax = plt.subplots()
        fig.suptitle(f"{variable}, {ydimension} vs. {xdimension} " + title_string)
        ax.set_xlabel(f"{xdimension}")
        ax.set_ylabel(f"{ydimension}")
        im = ax.imshow(
            data_slice, cmap=cmap, origin="lower", extent=extent, aspect="auto", **kwargs
        )
        plt.colorbar(mappable=im)
        return fig, ax

    def plot_cutoff_contour(
        self,
        const_angle_str,
        const_angle,
        save_path=None,
        measure="rho",
        mask_flag=True,
        bounds=None,
        return_artists=True,
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
            return_artists (bool, optional): Default True. If True, returns fig, ax used to
            plot the diagram, else returns the arrays used for plotting.

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

        if const_angle not in self.get_coordinate_array(const_angle_str):
            raise ValueError("Value of const_angle not in original coordinate array.")

        coords_dict = {const_angle_str: const_angle}
        xdimension = "frequency"
        ydimension = var_angle_str
        spline_surface = self.create_2Dspline(
            variable, xdimension, ydimension, coords_dict
        )

        x_coordinates = self.get_coordinate_array(xdimension)
        y_coordinates = self.get_coordinate_array(ydimension)

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

        x_array = np.linspace(x_start, x_stop, num=500)
        y_array = np.linspace(y_start, y_stop, num=500)
        Xgrid, Ygrid = np.meshgrid(x_array, y_array, indexing="ij")
        Zgrid = spline_surface((Xgrid, Ygrid))

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
            CS = ax.contourf(
                Xgrid, Ygrid, Z_masked, levels=20, cmap="plasma_r", corner_mask=True
            )
            CSgo = ax.contour(
                Xgrid, Ygrid, Z_masked, levels=20, colors='w', corner_mask=True, linewidth=0.2
            )
        else:
            CS = ax.contourf(
                Xgrid, Ygrid, Zgrid, levels=20, cmap="plasma_r", corner_mask=True
            )
            CSgo = ax.contour(
                Xgrid, Ygrid, Z_grid, levels=20, colors='w', corner_mask=True, linewidth=0.2
            )

        fig.suptitle(f"Cutoff {title_label}, {const_angle_str}={const_angle}$^\circ$")
        ax.set_xlabel("frequency/GHz")
        ax.set_ylabel(f"{var_angle_str}/$^\circ$")
        ax.clabel(CSgo, inline=True, fontsize=6)
        fig.colorbar(CS)

        if save_path:
            plt.savefig(
                save_path + f"cutoff_contour_{const_angle_str}{const_angle}.jpg",
                dpi=200,
            )

        if return_artists:
            return fig, ax
        return (Xgrid, Ygrid, Zgrid, interp_mask)

    def plot_cutoff_angles(self, poloidal_angle):  # TODO: Implement this
        return

    def plot_opt_tor_contour(self):
        freq = self.get_coordinate_array('frequency')
        pol = self.get_coordinate_array('poloidal_angle')
        x = np.linspace(freq[0], freq[-1], 500)
        y = np.linspace(pol[0], pol[-1], 500)
        X, Y = np.meshgrid(x, y, indexing='ij')
        spline = self.create_2Dspline(
            variable='opt_tor', 
            xdimension='frequency',
            ydimension='poloidal_angle',
            )
        data = spline((X, Y))
        mask_func = RegularGridInterpolator(
            points=(freq,pol), 
            values=self.dataset['opt_tor_mask'].transpose('frequency','poloidal_angle').values,
            method='nearest',
        )
        data_masked = np.ma.array(data, mask=mask_func((X, Y)))
        fig, ax = plt.subplots()

        CS = ax.contourf(
                X, Y, data_masked, levels=20, cmap="plasma_r", corner_mask=True
            )
        CSgo = ax.contour(
                X, Y, data_masked, levels=20, colors="w", corner_mask=True, linewidth=0.2
            )
        fig.suptitle("Opt. toroidal angle contour")
        ax.set_xlabel("frequency/GHz")
        ax.set_ylabel("poloidal angle/$^\circ$")
        ax.clabel(CSgo, inline=True, fontsize=6)
        fig.colorbar(CS)
        return fig, ax

    def plot_delta_m_contour(self, poloidal_angle):
        return

    # To be implemented
    """ 
    

    def plot_opt_tor_vs_freq(self, poloidal_angle):
        return
    """


class PitchDiagnostic:
    """Separate class for representing a particular pitch angle diagnostic design/setup.
    Analysis methods take in Scotty simulation data in a SweepDataset class, and saves the
    output in a dictionary of datasets. Each dataset corresponds to a different SweepDataset
    equilibriums. Datasets are organized into NetCDF groups when reading and writing to file.

    Class is constructed by calling class methods create_new or from_netcdf.


    """

    def __init__(self, ds):
        self.home = ds
        self.ds_dict = {}

    # I/O TODO: Fix lingerin I/O issues with variables and attributes not being written properly

    @classmethod
    def create_new(
        cls,
        poloidal_angles,
        toroidal_angles,
        frequencies,
        position=[],
        curvatures=[],
        std_noise=0.05,
    ):
        """Creates a new PitchDiagnostic instance by specifying a poloidal and toroidal array as
        well as a set of discrete frequencies representing the simulated diagnostic.

        Args:
            poloidal_angles (array): Poloidal steering positions for array of microwave antennae
            toroidal_angles (array): Toroidal steering positions for array of microwave antennae
            frequencies (array): Discrete frequency coverage of microwave antennaes
            poloidal_position (array, optional): (R, Z) coordinate launch positions of the antennae,
            toroidal angle coordinate is not required due to symmetry. Default None assumes that
            poloidal_position is identical to SweepDataset simulation parameters.
            curvatures (float/array, optional): Beam curvature for launch beam, if array must be the
            same length as frequencies. Default None assumes curvature is identical to SweepDataset
            simulation parameters.
            std_noise: Standard deviation of gaussian noise used to simulate antenna response
        """
        ds = xr.Dataset()
        ds.attrs["class_type"] = "PitchDiagnostic"
        ds.attrs["poloidal_angles"] = poloidal_angles
        ds.attrs["toroidal_angles"] = toroidal_angles
        ds.attrs["frequencies"] = frequencies
        ds.attrs["position"] = position
        ds.attrs["curvatures"] = curvatures
        ds.attrs["std_noise"] = std_noise
        classobject = cls(ds)

        return classobject

    @classmethod
    def from_netcdf(cls, path_string):
        """ """
        with xr.open_dataset(path_string, group="home") as ds:
            class_type = ds.attrs["class_type"]
            if class_type != "PitchDiagnostic":
                raise ValueError(
                    ".nc file does not have class_type = 'PitchDiagnostic' attribute"
                )
            classobject = cls(ds)
        classobject.ds_dict = {}
        # Deal with annoying beahviour where xarray squeezes any iterable attributes
        # with only one element
        ds_list = classobject.home.attrs["ds_list"]
        if type(ds_list) == str:
            ds_list = [ds_list]
        for key in ds_list:
            with xr.open_dataset(path_string, group=key) as ds:
                classobject.ds_dict[key] = ds

        return classobject

    def to_netcdf(self, folder="", filename="PitchDiagnostic", suffix=""):
        """Saves contents of Dataset into a netCDF4 file for easy read and writing.

        Args:
            folder (str, optional): Path to folder to save file to. Default is None.
            filename (str, optional): Filename to save the nc file as. Default is 'SweepDataset'.
        """
        file_path = folder + filename + suffix + ".nc"

        # Save ds_dict into home
        self.home.attrs["ds_list"] = [key for key in self.ds_dict.keys()]
        self.home.to_netcdf(path=file_path, group="home", format="NETCDF4")
        for key in self.home.attrs["ds_list"]:
            ds = self.ds_dict[key]
            ds.to_netcdf(path=file_path, group=key, mode="a", format="NETCDF4")

    ## define properties
    @property
    def poloidal_angles(self):
        poloidal_angles = self.home.attrs["poloidal_angles"]
        # Account for annoying behaviour where xarray automatically squeezes attributes
        try:
            output = poloidal_angles[0]
        except IndexError:
            poloidal_angles = np.array([poloidal_angles])
        return poloidal_angles

    @property
    def toroidal_angles(self):
        return self.home.attrs["toroidal_angles"]

    @property
    def frequencies(self):
        return self.home.attrs["frequencies"]

    @property
    def position(self):
        return self.home.attrs["position"]

    @property
    def curvatures(self):
        return self.home.attrs["curvatures"]

    @property
    def std_noise(self):
        return self.home.attrs["std_noise"]

    ## get methods

    def get_keys(self):
        keys_dict = {}
        for key in ds_dict.keys():
            values = ds_dict[key].keys()
            keys_dict[key] = tuple(values)
        print(keys)
        return keys_dict

    def get_DataArray(self):
        return

    ## Simulation methods
    def set_rho_freq_relation(self, SweepDataset): #Assumed equilibrium relation for analysing all equilibria
        # Use an array representation of a spline
        # 2D spline contour averaged over toroidal range to gSet average relation
        return

    def get_rho_shift(self, SweepDataset): # Maybe merge as part of previous method?
        return


    def get_opt_tor_delta(self, SweepDataset):
        descriptor = str(SweepDataset.descriptor)
        if descriptor in self.ds_dict:
            ds = self.ds_dict[descriptor]
            if "opt_tor" in ds.keys():
                print("opt_tor already retrieved!")
                return ds["opt_tor"]
        else:
            self.ds_dict[descriptor] = xr.Dataset()

        opt_tor_spline = SweepDataset.create_2Dspline(
            variable="opt_tor", xdimension="frequency", ydimension="poloidal_angle"
        )
        x, y = self.frequencies, self.poloidal_angles
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
        opt_tor_array = opt_tor_spline((x_grid, y_grid))
        opt_tor_da = xr.DataArray(
            data=opt_tor_array,
            dims=("frequency", "poloidal_angle"),
            coords={
                "frequency": x,
                "poloidal_angle": y,
            },
        )
        self.ds_dict[descriptor]["opt_tor"] = opt_tor_da

        delta_m_spline = SweepDataset.create_3Dspline(
            variable='cutoff_delta_theta_m',
            xdimension='frequency',
            ydimension='poloidal_angle',
            zdimension='toroidal_angle',
        )
        X, Y, Z = np.meshgrid(self.frequencies, self.poloidal_angles, self.toroidal_angles, indexing='ij')
        delta_m_array = delta_m_spline((X, Y, Z))
        delta_theta_m_da = xr.DataArray(
            data = delta_m_array,
            dims=("frequency", "poloidal_angle", "toroidal_angle"),
            coords={
                "frequency": self.frequencies,
                "poloidal_angle": self.poloidal_angles,
                "toroidal_angle": self.toroidal_angles
            },
        )
        self.ds_dict[descriptor]["delta_theta_m"] = delta_theta_m_da
        return opt_tor_da, delta_theta_m_da

    def simulate_measurements(self, SweepDataset, iterations=500, std_noise=None, noise_flag=True):
        """Simulates backscattered power with noise received by antenna array through the
        frequency range and stores it in a dataset, with the variable set by the SweepDataset
        descriptor. Data is saved with the key 'simulated_measurements'.

        Args:
            SweepDataset (SweepDataset): SweepDataset class object used to simulate power received
            by antenna
            iterations (int): Number of iterated measurements
            std_noise (float): Standard deviation of gaussian noise. Default is 'std_noise' attribute
            initialized in the class.
            noise_flag (bool): Turn noise on or off. True by default.
        """
        if not std_noise:
            std_noise = self.std_noise

        frequencies = np.array(self.frequencies)
        toroidal_angles = np.array(self.toroidal_angles)
        poloidal_angles = np.array(self.poloidal_angles)
        iter_array = np.arange(1, iterations + 1)
        toroidal_range = np.linspace(toroidal_angles[0], toroidal_angles[-1], 500)
        descriptor = str(SweepDataset.descriptor)

        if descriptor in self.ds_dict:
            sim_ds = self.ds_dict[descriptor]
            print(sim_ds.keys())
            if "simulated_measurements" in sim_ds.keys():
                print("Simulation data already present!")
                return sim_ds["simulated_measurements"]

            else:
                self.ds_dict[descriptor]["simulated_measurements"] = xr.DataArray(
                    dims=("frequency", "poloidal_angle", "toroidal_angle", "iteration"),
                    coords={
                        "frequency": frequencies,
                        "poloidal_angle": poloidal_angles,
                        "toroidal_angle": toroidal_angles,
                        "iteration": iter_array,
                    },
                )
                self.ds_dict[descriptor]["power_profile"] = xr.DataArray(
                    dims=("frequency", "poloidal_angle", "toroidal_range"),
                    coords={
                        "frequency": frequencies,
                        "poloidal_angle": poloidal_angles,
                        "toroidal_range": toroidal_range,
                    },
                )

        else:
            sim_ds = xr.Dataset()
            sim_ds["simulated_measurements"] = xr.DataArray(
                dims=("frequency", "poloidal_angle", "toroidal_angle", "iteration"),
                coords={
                    "frequency": frequencies,
                    "poloidal_angle": poloidal_angles,
                    "toroidal_angle": toroidal_angles,
                    "iteration": iter_array,
                },
            )
            sim_ds["power_profile"] = xr.DataArray(
                dims=("frequency", "poloidal_angle", "toroidal_range"),
                coords={
                    "frequency": frequencies,
                    "poloidal_angle": poloidal_angles,
                    "toroidal_range": toroidal_range,
                },
            )
            self.ds_dict[descriptor] = sim_ds

        da = self.ds_dict[descriptor]["simulated_measurements"]
        db = self.ds_dict[descriptor]["power_profile"]

        for frequency in frequencies:
            for poloidal_angle in poloidal_angles:
                simulated_power_prof = SweepDataset.get_simulated_power_prof(
                    frequency, poloidal_angle, std_noise=std_noise,
                )
                input_array = np.repeat(
                    toroidal_angles[:, np.newaxis], repeats=iterations, axis=1
                )
                input_slice = xr.DataArray(
                    input_array,
                    coords={
                        "toroidal_angle": toroidal_angles,
                        "iteration": iter_array,
                    },
                )
                output_slice = xr.apply_ufunc(
                    simulated_power_prof, input_slice, vectorize=True
                )
                index = {"frequency": frequency, "poloidal_angle": poloidal_angle}
                da.loc[index] = output_slice
                power_prof = simulated_power_prof(toroidal_range, noise_flag=False)
                db.loc[index] = power_prof

        self.ds_dict[descriptor]["simulated_measurements"] = da

        return da

    ## Analysis methods
    ## TODO: function to find range of shift in rho across a poloidal sweep
    ## compare the delta rho to the radial position measured by set of frequencies

    def fit_measurement_gaussians(self, descriptor=None):
        if descriptor:
            descriptors = [descriptor]
        else:
            descriptors = self.ds_dict.keys()

        for descriptor in descriptors:
            if "gaussian_fit_coeffs" in self.ds_dict[descriptor]:
                print(f"Gaussian fits already generated for {descriptor}")
                continue

            data = self.ds_dict[descriptor]["simulated_measurements"]
            curvefit_results = data.curvefit(coords="toroidal_angle", func=fit_gaussian)
            curvefit_results["curvefit_coefficients"].attrs[
                "method"
            ] = "PitchDiagnostic.fit_gaussians"
            curvefit_results["curvefit_coefficients"].attrs["descriptor"] = descriptor
            curvefit_results["curvefit_coefficients"].attrs["fit_parameters"] = (
                "opt_tor",
                "delta_theta_m",
                "amplitude",
            )
            self.ds_dict[descriptor]["gaussian_fit_coeffs"] = curvefit_results[
                "curvefit_coefficients"
            ]
            self.ds_dict[descriptor]["gaussian_fit_cov"] = curvefit_results[
                "curvefit_covariance"
            ]

    def aggregate_fitted_gaussians(self, descriptor=None):
        if descriptor:
            descriptor = [descriptor]
        else:
            descriptors = self.ds_dict.keys()

        for descriptor in descriptors:
            if "mean_opt_tor" in self.ds_dict[descriptor]:
                print(f"Gaussian fits already aggregated for {descriptor}")
                continue

            coefficients = self.ds_dict[descriptor]["gaussian_fit_coeffs"]
            mean_opt_tor = coefficients.mean(dim="iteration", skipna=True)
            std_opt_tor = coefficients.std(dim="iteration", skipna=True, ddof=1)
            self.ds_dict[descriptor]["mean_opt_tor"] = mean_opt_tor.isel(param=0)
            self.ds_dict[descriptor]["std_opt_tor"] = std_opt_tor.isel(param=0)
            self.ds_dict[descriptor]["mean_delta_theta_m"] = mean_opt_tor.isel(param=1)
            self.ds_dict[descriptor]["std_delta_theta_m"] = std_opt_tor.isel(param=1)

    ## Plot methods
    def plot_power_profile(self, descriptor):
        data = self.ds_dict[descriptor]["power_profile"]
        opt_tor = self.ds_dict[descriptor]["mean_opt_tor"].loc[{"poloidal_angle": self.poloidal_angles[0]}]
        delta = self.ds_dict[descriptor]["mean_delta_theta_m"].loc[{"poloidal_angle": self.poloidal_angles[0]}]
        actual = self.ds_dict[descriptor]["opt_tor"].loc[{"poloidal_angle": self.poloidal_angles[0]}]

        toroidal_range = data.coords["toroidal_range"].values
        colormap = plt.cm.gnuplot2
        freq_count = len(self.frequencies)
        cols = None
        rows = None
        for number in (2, 3, 4, 5):
            if freq_count % number == 0:
                cols = number
                rows = freq_count // number
                break
        if not cols:
            cols = 1
            rows = freq_count

        combinations = list(product(range(rows), range(cols)))
        counter = 0
        fig, axes = plt.subplots(rows, cols, dpi=150, figsize=(5, 7), sharex=True, sharey=True)
        fig.tight_layout()

        for frequency in self.frequencies:
            row, col = combinations[counter]
            ax = axes[row, col]
            color = 'k'

            lin_data = data.loc[{'frequency':frequency, 'poloidal_angle':self.poloidal_angles[0]}]
            opt_tor_val = opt_tor.loc[{'frequency':frequency}].values
            delta_val = delta.loc[{'frequency':frequency}].values
            gaussian_profile = fit_gaussian(toroidal_range, opt_tor_val, delta_val, 1.0)

            ax.plot(toroidal_range, lin_data, alpha=0.5, color=color, label='profile')
            ax.plot(toroidal_range, gaussian_profile, alpha=0.5, color='r', linestyle='--', label='gaussian fit')
            ax.axvline(opt_tor_val, label='mean gaussian-fitted peak', color='r', linestyle='--')
            ax.axvline(actual.loc[{'frequency':frequency}], label='predicted opt_tor', color=color)
            ax.set_title(f"{frequency} GHz", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=5)

            #ax.legend()
            counter += 1

        ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0., fontsize=8)
        fig.suptitle("Power Profile vs. Toroidal Angle", fontsize=15)
        plt.subplots_adjust(top=0.90)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('toroidal steering/deg')
        plt.ylabel('normalized received power')
        return fig, axes

    def plot_delta_m(self, descriptor):
        data = self.ds_dict[descriptor]["delta_theta_m"]
        actual = self.ds_dict[descriptor]["opt_tor"].loc[{"poloidal_angle": self.poloidal_angles[0]}]

        freq_count = len(self.frequencies)
        cols = None
        rows = None
        for number in (2, 3, 4, 5):
            if freq_count % number == 0:
                cols = number
                rows = freq_count // number
                break
        if not cols:
            cols = 1
            rows = freq_count

        combinations = list(product(range(rows), range(cols)))
        counter = 0
        fig, axes = plt.subplots(rows, cols, dpi=150, figsize=(5, 7), sharex=True, sharey=True)
        fig.tight_layout()

        for frequency in self.frequencies:
            row, col = combinations[counter]
            ax = axes[row, col]
            
            ax.plot(self.toroidal_angles, data.loc[{'frequency':frequency, 'poloidal_angle':self.poloidal_angles[0]}], label='mismatch tolerance')
            ax.axvline(actual.loc[{'frequency':frequency}], label='opt_tor', color='k')
            ax.set_title(f"{frequency} GHz", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=5)

            counter += 1

        ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0., fontsize=8)
        fig.suptitle("Mismatch tolerance vs. Toroidal Steering", fontsize=15)
        plt.subplots_adjust(top=0.90)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('toroidal steering/deg')
        plt.ylabel('normalized received power')
        return fig, axes

    def plot_opt_tor_hist(self, descriptor):

        mean = self.ds_dict[descriptor]["mean_opt_tor"].loc[{"poloidal_angle": self.poloidal_angles[0]}]
        actual = self.ds_dict[descriptor]["opt_tor"].loc[{"poloidal_angle": self.poloidal_angles[0]}]

        colormap = plt.cm.gnuplot2

        freq_count = len(self.frequencies)
        cols = None
        rows = None
        for number in (2, 3, 4, 5):
            if freq_count % number == 0:
                cols = number
                rows = freq_count // number
                break
        if not cols:
            cols = 1
            rows = freq_count

        data = self.ds_dict[descriptor]["gaussian_fit_coeffs"].isel(param=0)
        counter = 0
        combinations = list(product(range(rows), range(cols)))

        fig, axes = plt.subplots(rows, cols, dpi=150, figsize=(5, 7), sharex=True, sharey=True)
        fig.tight_layout()
        for frequency in self.frequencies:
            row, col = combinations[counter]
            ax = axes[row, col]
            data_slice = data.loc[
                {"frequency": frequency, "poloidal_angle": self.poloidal_angles[0]}
            ]
            color = colormap(counter / freq_count)
            ax.hist(data_slice, alpha=0.5, color=color)
            ax.axvline(mean.loc[{'frequency':frequency}], label='mean', color='k')
            ax.axvline(actual.loc[{'frequency':frequency}], label='predicted', color='r', linestyle='--')
            ax.set_title(f"{frequency} GHz", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=5)
            
            #ax.legend()
            counter += 1
        ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0., fontsize=8)
        fig.suptitle("noisy opt_tor fits", fontsize=15)
        plt.subplots_adjust(top=0.90)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('Measured optimum toroidal steering/deg')
        plt.ylabel('Number of measurements')

        return fig, axes

    def plot_opt_tor_vs_freq(self, descriptor): #TODO: Plot line + std error bars + actual measurements
            mean = self.ds_dict[descriptor]["mean_opt_tor"].loc[{"poloidal_angle": self.poloidal_angles[0]}]
            std = self.ds_dict[descriptor]["std_opt_tor"].loc[{"poloidal_angle": self.poloidal_angles[0]}]
            actual = self.ds_dict[descriptor]["opt_tor"].loc[{"poloidal_angle": self.poloidal_angles[0]}]
            x_coords = self.frequencies
            plt.figure()
            plt.title('Opt Tor vs. Probe Frequency')
            plt.plot(x_coords, mean, label='mean', marker='+')
            plt.fill_between(x_coords, mean-std, mean+std, label='1$\sigma$', alpha=0.3)
            plt.plot(x_coords, actual, label='predicted', color='r', marker='+')
            plt.xlabel('Frequency/GHz')
            plt.ylabel('Opt. Toroidal Angle/Deg')
            plt.legend()
            return plt

    def plot_opt_tor_vs_rho(self): #TODO: Plot against rho; need to separate rho-freq relation based
        # on theoretical equilibrium (should be an attribute)
            return

# Helper functions


def fit_gaussian(toroidal_angle, opt_tor, delta, amplitude):
    return amplitude * np.exp(-(((toroidal_angle - opt_tor) / delta) ** 2))


def gaussian(theta_m, delta):
    return np.exp(-((theta_m / delta) ** 2))


def noisy_gaussian(theta_m, delta, std=0.05):
    mean = gaussian(theta_m, delta)
    return mean + np.random.normal(mean, std)


def find_nearest(array, value):
    index = np.abs((array - value)).argmin()
    return array[index]
