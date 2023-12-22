"""Class to load poloidal + toroidal + frequency sweeps of 
Scotty output data. Uses xarrays with poloidal angle, toroidal
angle and frequency as axes. 

"""

# Copyright 2023, Valerian Hall-Chen and Scotty contributors
# SPDX-License-Identifier: GPL-3.0


import xarray as xr
import DataTree as dt
import numpy as np
import json
import warnings
from scipy.interpolate import (
    UnivariateSpline,
    RectBivariateSpline,
    RegularGridInterpolator,
    splrep,
    sproot,
    PPoly,
    Akima1DInterpolator,
)
from scipy.optimize import newton, root_scalar, minimize_scalar, basinhopping
import matplotlib.pyplot as plt
import matplotlib as mpl
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
        self.topfile = None

    # I/O
    @classmethod
    def from_DataTree(
        cls,
        input_path,
        frequencies,
        poloidal_angles,
        toroidal_angles,
        filepath_format,
        attrs_dict={},
    ):
        """Constructs a SweepDataset instance from a set of Scotty .h5 output files.

        Args:
            input_path (string): Path to home folder containing all outputs
            frequencies (iterable): Array or List of indexing frequencies
            toroidal_angles (iterable): Array or List of indexing toroidal launch angles
            poloidal_angles (iterable): Array or List of indexing poloidal launch angles
            filepath_format (f-string): an f-string that specifies the path string to be
            appended to the end of the input_path to access outputs of individual runs.
            Note that the file path or filenames must include {frequency}, {toroidal_angle},
            and {poloidal_angle} specified in frequencies, toroidal_angles, and poloidal_angles.
            attrs_dict (dictionary): A dictionary of attribute names and values to save to
            the xarray dataset.
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
            "loc_m",
            "loc_b",
            "loc_p",
            "loc_r",
        ]

        print("Reading .h5 files...")

        for variable in variables:
            ds[variable] = xr.DataArray(
                coords=[
                    ("frequency", frequencies),
                    ("poloidal_angle", poloidal_angles),
                    ("toroidal_angle", toroidal_angles),
                ]
            )

        for combination in product(frequencies, toroidal_angles, poloidal_angles):
            frequency, toroidal_angle, poloidal_angle = combination
            index = {
                "frequency": float(frequency),
                "poloidal_angle": float(poloidal_angle),
                "toroidal_angle": float(toroidal_angle),
            }

            # File paths for the scotty .h5 output file
            path = input_path + filepath_format.format(
                frequency=frequency,
                poloidal_angle=poloidal_angle,
                toroidal_angle=toroidal_angle,
            )

            # Load data into the empty Dataset

            input_dict = {
                "launch_position": "launch_position",
                "mode_flag": "mode_flag",
                "data_R_coord": "R",
                "data_Z_coord": "Z",
                "ne_data_density_array": "ne_data_density_array",
                "ne_data_radialcoord_array": "ne_data_radialcoord_array",
            }

            try:
                with dt.open_datatree(path) as tree:
                    analysis = tree["analysis"]
                    path_len = len(tree["distance_along_line"].values)

                    attribute_dict = {
                        "distance_along_line": "distance_along_line",
                        "delta_theta_m": "delta_theta_m",
                        "theta_m_output": "theta_m",
                        "theta_output": "theta",
                        "K_magnitude_array": "K_magnitude",
                        "loc_m": "loc_m",
                        "loc_b": "loc_b",
                        "loc_p": "loc_p",
                        "loc_r": "loc_r",
                        "poloidal_flux_output": "poloidal_flux",
                    }

                    for attribute in (
                        "distance_along_line",
                        "delta_theta_m",
                        "theta_m_output",
                        "theta_output",
                        "K_magnitude_array",
                        "loc_m",
                        "loc_b",
                        "loc_p",
                        "loc_r",
                        "poloidal_flux_output",
                    ):
                        key = attribute_dict[attribute]
                        current_ds = ds[attribute]
                        data = analysis[key].values
                        if len(current_ds.dims) < 4:
                            ds[attribute] = current_ds.expand_dims(
                                dim={"trajectory_step": np.arange(0, path_len)}, axis=-1
                            ).copy()
                        ds[attribute].loc[index] = data

                    ds["cutoff_index"].loc[index] = analysis["cutoff_index"].values

                    output = tree["solver_output"]
                    attribute_dict2 = {
                        "q_R_array": "q_R",
                        "q_zeta_array": "q_zeta",
                        "q_Z_array": "q_Z",
                        "K_R_array": "K_R",
                        "K_Z_array": "K_Z",
                    }
                    for attribute in (
                        "q_R_array",
                        "q_zeta_array",
                        "q_Z_array",
                        "K_R_array",
                        "K_Z_array",
                    ):
                        key = attribute_dict2[attribute]
                        current_ds = ds[attribute]
                        if "trajectory_step" not in current_ds.dims:
                            ds[attribute] = current_ds.expand_dims(
                                dim={"trajectory_step": np.arange(0, path_len)}, axis=-1
                            ).copy()
                        ds[attribute].loc[index] = output[key]

                    ds["K_zeta_initial"].loc[index] = tree["inputs"][
                        "K_initial"
                    ].values[1]

            except FileNotFoundError:
                print(
                    f"No file found for freq={frequency} pol={poloidal_angle}, tor={toroidal_angle}"
                )
                missing_indices.append(index)
                continue

        for combination in product(frequencies, toroidal_angles, poloidal_angles):
            frequency, toroidal_angle, poloidal_angle = combination
            index = {
                "frequency": float(frequency),
                "poloidal_angle": float(poloidal_angle),
                "toroidal_angle": float(toroidal_angle),
            }
            # File paths for the scotty .h5 output file
            path = input_path + filepath_format.format(
                frequency=frequency,
                poloidal_angle=poloidal_angle,
                toroidal_angle=toroidal_angle,
            )

            try:
                with dt.open_datatree(path) as tree:
                    input_file = tree["inputs"]
                    for attribute in input_dict.keys():
                        key = input_dict[attribute]
                        ds.attrs[attribute] = input_file[attribute]

                    anaysis = tree["analysis"]
                    ds.attrs["poloidal_flux_on_midplane"] = analysis[
                        "poloidal_flux_on_midplane"
                    ]
                    ds.attrs["R_midplane_points"] = analysis["R_midplane"]
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
        return classobject

    @classmethod
    def from_netcdf(cls, path_string):
        """Constructs a SweepDataset instance from an exported netCDF file.

        Args:
            path_string (str): Path to .nc file

        """
        with xr.load_dataset(path_string) as ds:
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
        attrs_dict={},
    ):
        """Constructs a SweepDataset instance from a set of Scotty .npz output files.

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
            attrs_dict (dictionary): A dictionary of attribute names and values to save to
            the xarray dataset.
        """
        output_types = ("analysis_output", "data_input", "data_output", "solver_output")

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
            "loc_m",
            "loc_b",
            "loc_p",
            "loc_r",
        ]

        input_attributes = [
            "launch_position",
            "mode_flag",
            "data_R_coord",
            "data_Z_coord",
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
            frequency, toroidal_angle, poloidal_angle = combination
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
                        "loc_m",
                        "loc_b",
                        "loc_p",
                        "loc_r",
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
                        if "trajectory_step" not in current_ds.dims:
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
            frequency, toroidal_angle, poloidal_angle = combination
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
                    ds.attrs["R_midplane_points"] = analysis_file["R_midplane_points"]
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
        return classobject

    def interpolate_missing_values(self):
        """Attempts to interpolate missing data in cutoff indices and float variable DataArrays."""
        print("Current gaps in data:")
        for item in self.missing_indices:
            print(item)
        print("Interpolating missing values...")
        output1 = self.check_cutoff_indices()
        output2 = self.check_all_float_arrays()
        print(
            "Interpolation complete. Check print output for unsuccessful interpolations."
        )

    def to_netcdf(self, folder="", filename=None):
        """Saves contents of Dataset into a netCDF4 file for easy read and writing.

        Args:
            folder (str, optional): Path to folder to save file to. Default is an empty string.
            filename (str, optional): Filename to save the nc file as. Default is 'SweepDataset', or
            the descriptor string if provided.
        """
        self.dataset.attrs["missing_indices"] = json.dumps(self.missing_indices)
        filename = filename or self.dataset.get("descriptor", "SweepDataset")
        file_path = f"{folder}{filename}.nc"
        self.dataset.to_netcdf(path=file_path)
        print(f"File saved to {file_path}")

    #### Set Methods ####

    def set_attrs(self, attribute, value):
        """Used to set attributes of the xarray dataset.

        Args:
            attribute (str): Name of the attribute.
            value (hashable): Value of the attribute.
        """
        self.dataset.attrs[attribute] = value

    def set_descriptor(self, descriptor):
        """Used to set a unique descriptor to identify sweeps with different plasma
        equilibrium. This descriptor is needed to tag SweepDataset data when loaded
        onto a PitchDiagnostic object.

        Args:
            descriptor (hasable): Typically a string to describe the SweepDataset,
            but it is possible to use floats or integers as well.
        """
        self.set_attrs("descriptor", descriptor)

    def set_topfile_path(self, topfile_path):
        """Sets the path string to the json topfile associated with this set of
        SweepDataset data. This path string can thus be saved to persistent
        memory in netcdf format and be read again.

        Args:
            topfile_path (str): Path string to the json topfile.
        """
        if "topfile_path" in self.dataset.attrs.keys():
            warnings.warn("Overwriting previously set topfile path.")
        self.dataset.attrs["topfile_path"] = topfile_path

    def get_topfile_path(self):
        """Get method for the topfile path attribute."""
        if "topfile_path" in self.dataset.attrs.keys():
            return self.dataset.attrs["topfile_path"]
        raise ValueError(f"No topfile path set.")

    def load_topfile(self):
        """Loads data from the topfile associated with the topfile_path string and
        saves it to a separate self.topfile xarray dataset. This dataset is not saved
        to the .nc file when the to_netcdf method is called since it is already saved
        in persistent memory in json format.

        Returns:
            Dataset: An xarray Dataset of the json data.
        """
        topfile_path = self.get_topfile_path()
        ds = xr.Dataset()
        with open(topfile_path) as f:
            topfile = json.load(f)
            R_coord = topfile["R"]
            Z_coord = topfile["Z"]
            B_R = np.transpose(topfile["Br"])
            B_Z = np.transpose(topfile["Bz"])
            B_T = np.transpose(topfile["Bt"])
            pol_flux = np.transpose(topfile["pol_flux"])
        var_names = ("Br", "Bz", "Bt", "pol_flux")
        arrays = (B_R, B_Z, B_T, pol_flux)
        for i in range(len(arrays)):
            array = arrays[i]
            var_name = var_names[i]
            ds[var_name] = xr.DataArray(
                data=array,
                dims=("R", "Z"),
                coords={
                    "R": R_coord,
                    "Z": Z_coord,
                },
            )
        # Find pitch angle
        ds["pitch_angle"] = np.rad2deg(
            np.arctan(
                np.divide(
                    np.hypot(ds["Br"], ds["Bz"]),
                    ds["Bt"],
                )
            )
        )
        self.topfile = ds
        return ds

    ## get methods/properties

    def get_Dataset_copy(self):
        """
        Returns:
            Dataset: Deep copy of the underlying xarray Dataset
        """
        return self.dataset.copy(deep=True)

    def get_coordinate_array(self, dimension):
        """
        Args:
            dimension (str): Name of the dimension.
        Returns:
            numpy array: Coordinate array associated with dimension.
        """
        data = self.dataset
        return data.coords[dimension].values

    def get_rho_freq_spline(self, poloidal_angle, toroidal_angle):
        """Method for retrieving a spline of cutoff rho as a function of frequency.
        Args:
            poloidal_angle (int/float): Poloidal angle to interpolate the spline at
            toroidal_angle (int/float): Toroidal angle to interpolate the spline at
        Returns:
            UnivariateSpline object
        """
        spline = self.create_1Dspline(
            variable="cutoff_rho",
            dimension="frequency",
            coords={
                "poloidal_angle": poloidal_angle,
                "toroidal_angle": toroidal_angle,
            },
        )
        return spline

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
        """A list of variable names indexing the component DataArrays of self.dataset."""
        return list(self.dataset.keys())

    @property
    def descriptor(self):
        """The descriptor string associated with this SweepDataset."""
        return str(self.dataset.attrs["descriptor"])

    def view(self):
        """Prints a summary of the objects and data contained within self.dataset."""
        print(self.dataset.keys())

    def generate_variable_at_cutoff(self, variable):
        """Finds the value of a given variable at cutoff and saves it with
        the key 'cutoff_{variable name}'.

        Args:
            variable:

        Returns:
            DataArray: A 3-D DataArray of cutoff distances with frequency,
            poloidal angle and toroidal angle as axes. Saves it with the
            key 'cutoff_distance'.
        """
        data = self.dataset

        if f"cutoff_{variable}" in self.variables:
            print("cutoff_{variable} already generated.")
            return data[f"cutoff_{variable}"]

        CUTOFF_ARRAY = data["cutoff_index"]
        cutoff_variable = data[variable].isel(trajectory_step=CUTOFF_ARRAY)
        self.dataset[f"cutoff_{variable}"] = cutoff_variable
        return cutoff_variable

    def generate_cutoff_distance(self):
        """Finds the poloidal cutoff distance of the beam from the
        magnetic axis. Saves a new 'poloidal_distance' 4-D DataArray for
        poloidal distance along trajectory in addition to cutoff distance.

        Returns:
            DataArray: A 3-D DataArray of cutoff distances with frequency,
            poloidal angle and toroidal angle as axes. Saves it with the
            key 'cutoff_distance'.
        """

        data = self.dataset
        B_axis = data.attrs["magnetic_axis_RZ"]
        if "cutoff_distance" in self.variables:
            print("cutoff_distance already generated.")
            return data["cutoff_index"]

        CUTOFF_ARRAY = data["cutoff_index"]
        # print(CUTOFF_ARRAY.shape)
        poloidal_distance = np.hypot(
            data["q_R_array"] - B_axis[0], data["q_Z_array"] - B_axis[1]
        )
        self.dataset["poloidal_distance"] = poloidal_distance
        # print(poloidal_distance.shape)
        cutoff_distance = poloidal_distance.isel(trajectory_step=CUTOFF_ARRAY)
        self.dataset["cutoff_distance"] = cutoff_distance
        return cutoff_distance

    def generate_cutoff_rho(self):
        """Finds the poloidal cutoff rho (sqrt flux) of the beam.

        Returns:
            DataArray: A 3-D DataArray of cutoff distances with frequency,
            poloidal angle and toroidal angle as axes. Saves it with the
            key 'cutoff_rho'.
        """
        data = self.dataset
        if "cutoff_rho" in self.variables:
            print("cutoff-rho already generated.")
            return data["cutoff_rho"]

        CUTOFF_ARRAY = data["cutoff_index"]
        cutoff_flux = data["poloidal_flux_output"].isel(trajectory_step=CUTOFF_ARRAY)
        self.dataset["cutoff_rho"] = np.sqrt(cutoff_flux)
        return self.dataset["cutoff_rho"]

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

    def generate_opt_tor(self):
        self.generate_mismatch_opt_tor()
        self.generate_loc_opt_tor()
        for key in ["opt_tor_mismatch", "opt_tor_loc_m", "opt_tor_loc_product"]:
            index_list = self.check_float_arrays(variable=key, method="nearest")
            print(f"{key} checked. Remaining indices:" + str(index_list))

    def generate_loc_opt_tor(self):
        for key in ["loc_m", "loc_product"]:
            if f"opt_tor_{key}" in self.variables:
                print(f"opt_tor_{key} already generated.")
                continue

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
                    coords = {
                        "frequency": frequency,
                        "poloidal_angle": poloidal_angle,
                    }
                    spline = self.create_1Dspline(
                        variable=f"int_{key}",
                        dimension="toroidal_angle",
                        coords=coords,
                        method="akima",
                    )
                    # derivative = spline.derivative()

                    try:
                        optimize_results = minimize_scalar(
                            fun=lambda x: -spline(x),
                            # x0=0,
                            bounds=(toroidal_angles[0], toroidal_angles[-1]),
                            method="bounded",
                        )

                        if optimize_results.success:
                            opt_tor_array.loc[coords] = np.squeeze(optimize_results.x)
                        else:
                            print(
                                f"No opt_tor found for Freq={frequency} GHz, Pol={poloidal_angle} deg"
                            )
                            opt_tor_array.loc[coords] = np.nan

                        array = np.linspace(-10, 10, 50)
                        plt.plot(array, spline(array))

                    except Exception as error:
                        print(
                            f"No zero found for Freq={frequency} GHz, Pol={poloidal_angle} deg: ",
                            error,
                        )
                        continue

            no_opt_tor_mask = opt_tor_array.isnull()
            self.dataset[f"opt_tor_{key}"] = opt_tor_array

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
            # self.dataset["nearest_opt_tor_index"] = closest_tor_array
            self.dataset[f"opt_tor_{key}_mask"] = np.logical_and(
                no_opt_tor_mask, reflected_opt_tor_mask
            )

    def generate_mismatch_opt_tor(self):
        """Uses scipy.optimize.newton on an interpolated spline to find
        the optimum toroidal steering with varying frequency and poloidal
        steering and saves it with the key 'opt_tor'.

        Returns:
            DataArray: A 2-D DataArray of optimal toroidal steering angles (in degrees)
        """
        if "opt_tor_mismatch" in self.variables:
            print("opt_tor_mismatch already generated.")

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
        # theta_m_array = self.dataset["cutoff_theta_m"]
        for frequency in frequencies:
            for poloidal_angle in poloidal_angles:
                coords = {
                    "frequency": frequency,
                    "poloidal_angle": poloidal_angle,
                }
                # x = toroidal_angles
                # y = theta_m_array.loc[coords].values
                # spl = splrep(x, y, s=0)
                # print(spl)
                spline = self.create_1Dspline(
                    variable="cutoff_theta_m",
                    dimension="toroidal_angle",
                    coords=coords,
                )
                try:
                    # print('flag 1')
                    # ppoly = PPoly.from_spline(spl)
                    # print(ppoly)
                    # roots = ppoly.roots()
                    # print('flag 3')
                    rootresults = root_scalar(
                        f=spline,
                        method="bisect",
                        bracket=(toroidal_angles[0], toroidal_angles[-1]),
                        x0=0,
                        x1=3,
                        maxiter=200,
                    )
                    # if len(roots) != 1:
                    #    print(f"Wrong number of roots found for Freq={frequency} GHz, Pol={poloidal_angle} deg. Roots: {roots}")
                    root = rootresults.root
                    opt_tor_array.loc[coords] = root
                except Exception as error:
                    print(
                        f"No zero found for Freq={frequency} GHz, Pol={poloidal_angle} deg: ",
                        error,
                    )
                    continue

        no_opt_tor_mask = opt_tor_array.isnull()
        self.dataset["opt_tor_mismatch"] = opt_tor_array

        def vfunc(arg):
            func = lambda x: find_nearest(toroidal_angles, x)
            return xr.apply_ufunc(func, arg, vectorize=True)

        closest_tor_array = vfunc(opt_tor_array)
        reflection_mask = self.generate_reflection_mask().transpose(
            "frequency", "poloidal_angle", "toroidal_angle"
        )
        reflected_opt_tor_mask = reflection_mask.isel(toroidal_angle=closest_tor_array)
        # self.dataset["nearest_opt_tor_index"] = closest_tor_array
        self.dataset["opt_tor_mismatch_mask"] = np.logical_and(
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
        return self.dataset["reflection_mask"]

    def _path_integrate(self, int_array):
        """Internal method used to integrate localization along path."""
        frequencies = self.get_coordinate_array("frequency")
        poloidal_angles = self.get_coordinate_array("poloidal_angle")
        toroidal_angles = self.get_coordinate_array("toroidal_angle")
        output_da = xr.DataArray(
            coords=[
                ("frequency", frequencies),
                ("poloidal_angle", poloidal_angles),
                ("toroidal_angle", toroidal_angles),
            ]
        )
        path_lengths = self.dataset["distance_along_line"]
        for combination in product(frequencies, poloidal_angles, toroidal_angles):
            frequency, poloidal_angle, toroidal_angle = combination
            index = {
                "frequency": float(frequency),
                "poloidal_angle": float(poloidal_angle),
                "toroidal_angle": float(toroidal_angle),
            }
            int_path = int_array.loc[index]
            path_length = path_lengths.loc[index]
            try:
                int_spline = UnivariateSpline(path_length, int_path, s=0)
                int_value = int_spline.integral(path_length[0], path_length[-1])
                output_da.loc[index] = int_value
            except ValueError as error:
                print(error)
                print("Frequency, poloidal_angle, toroidal_angle is:", combination)
        return output_da

    def generate_mismatch_gaussian(self):
        """Generates the backscattered power for each simulated run with the mismatch
        gaussian formula using values at cutoff.
        """
        if "mismatch_gaussian" in self.variables:
            print("mismatch_gaussian already generated.")
            return self.dataset["mismatch_gaussian"]
        delta_m = self.dataset["cutoff_delta_theta_m"]
        theta_m = self.dataset["cutoff_theta_m"]
        gaussian = np.exp(-((theta_m / delta_m) ** 2))
        self.dataset["mismatch_gaussian"] = gaussian
        return gaussian

    def integrate_loc_m(self):
        """Generates the backscattered power for each simulated run by integrating
        mismatch localization along the ray trajectory.
        """
        if "int_loc_m" in self.variables:
            print("int_loc_m already generated.")
            return self.dataset["int_loc_m"]
        loc_m = self.dataset["loc_m"]
        int_loc_m = self._path_integrate(loc_m)
        self.dataset["int_loc_m"] = int_loc_m
        return int_loc_m

    def integrate_loc_product(self):
        """Generates the backscattered power for each simulated run by integrating
        total localization (consisting of the ray, beam, polarization and mismatch
        pieces) along the ray trajectory.
        """
        if "int_loc_product" in self.variables:
            print("int_loc_product already generated.")
            return self.dataset["int_loc_product"]
        loc_m = self.dataset["loc_m"]
        loc_b = self.dataset["loc_b"]
        loc_p = self.dataset["loc_p"]
        loc_r = self.dataset["loc_r"]
        loc_product = loc_m * loc_b * loc_p * loc_r
        self.dataset["loc_product"] = loc_product
        int_loc_product = self._path_integrate(loc_product)
        self.dataset["int_loc_product"] = int_loc_product
        return int_loc_product

    def generate_all(self):
        """Calls all analysis methods to generate and save analysis data.
        Returns a list of variables including newly-generated ones that
        can be indexed.
        """
        self.generate_cutoff_distance()
        self.generate_cutoff_rho()
        self.generate_mismatch_at_cutoff()
        for variable in ("K_magnitude_array", "q_R_array", "q_Z_array"):
            self.generate_variable_at_cutoff(variable)
        self.integrate_loc_m()
        self.integrate_loc_product()
        self.generate_opt_tor()
        self.generate_reflection_mask()
        self.generate_mismatch_gaussian()

        return self.variables

    def generate_pitch_at_cutoff(self):
        """Determines the pitch angle at the cutoff point of the ray using
        data from the json topfile as the ground truth.
        """
        cutoff_R = self.dataset["cutoff_q_R_array"]
        cutoff_Z = self.dataset["cutoff_q_Z_array"]
        poloidal_angles = self.get_coordinate_array("poloidal_angle")
        frequencies = self.get_coordinate_array("frequency")
        toroidal_angles = self.get_coordinate_array("toroidal_angle")
        pitch = self.topfile["pitch_angle"].expand_dims(
            dim={
                "frequency": frequencies,
                "poloidal_angle": poloidal_angles,
                "toroidal_angle": toroidal_angles,
            },
        )
        cutoff_pitch = pitch.sel(R=cutoff_R, Z=cutoff_Z, method="nearest")
        self.dataset["cutoff_pitch"] = cutoff_pitch
        return cutoff_pitch

    ## simulation methods
    ## TODO: Future plans to accomodate simulation with integrated loc_m or total_loc; SweepDataset
    ## should provide the methods to simulate either using the mismatch gaussian formulae or by
    ## integrating localization, but PitchDiagnostic should be agnostic towards the method used to
    ## simulate the antenna response and run the same set of analyses on it.

    ## TODO: There is a bug where the simulated power profile gets sharply inverted crossing tor=0,
    ## likely due to a sqrt sign flip somewhere.

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
            method="pchip",
        )
        delta_spline = self.create_3Dspline(
            variable="cutoff_delta_theta_m",
            xdimension="frequency",
            ydimension="poloidal_angle",
            zdimension="toroidal_angle",
            method="pchip",
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

    def create_1Dspline(
        self, variable, dimension, coords={}, order=3, method="polynomial"
    ):
        """Memoized function that interpolates splines for any variable along
        a single arbitrary axis with scipy.interpolate.UnivariateSpline.

        Args:
            variable (str): index the specific DataArray to fit the spline to
            dimension (str): dimension name of the x-coordinate
            coords (dict): a dictionary of the form {dimension: value} for all
            other dimensions apart from the x-coordinate dimension. For example, if
            the spline is for interpolating theta_m as a function of toroidal_angle,
            coordinate values for all other dimensions of theta_m (frequency and
            poloidal angle) must be supplied.

        Returns:
            Callable: An interpolated UnivariateSpline function.
        """
        coords_hashable = self._dict_to_hashable(coords)
        args = (variable, dimension, coords_hashable)
        if args in self.spline_memo.keys():
            return self.spline_memo[args]
        else:
            data = self.dataset[variable]
            x_coordinates = self.get_coordinate_array(dimension)
            y_data = data.loc[coords]
            y_values = y_data.values
            if method == "polynomial":
                spline = UnivariateSpline(x=x_coordinates, y=y_values, s=0, k=order)
            elif method == "akima":
                spline = Akima1DInterpolator(x=x_coordinates, y=y_values)
            self.spline_memo[args] = spline
            return spline

    def create_2Dspline(
        self, variable, xdimension, ydimension, coords={}, method="linear"
    ):
        """Memoized function that interpolates splines for any variable along
        two arbitrary axes with linear interpolation (scipy RegularGridInterpolator).

        Args:
            variable (str): index the specific DataArray to fit the spline to
            xdimension (str): dimension name of the x-coordinate
            ydimension (str): dimension name of the y-coordinate
            coords (dict): a dictionary of the form {dimension: value} for all
            other dimensions apart from the x and y coordinate dimensions. See
            create_1Dspline for a similar method.

        Returns:
            Callable: A RegularGridInterpolator function.
        """
        coords_hashable = self._dict_to_hashable(coords)
        args = (variable, xdimension, ydimension, coords_hashable)
        if args in self.spline_memo.keys():
            return self.spline_memo[args]
        else:
            data = self.dataset[variable]
            x_coordinates = self.get_coordinate_array(xdimension)
            y_coordinates = self.get_coordinate_array(ydimension)
            z_values = data.loc[coords].transpose(xdimension, ydimension).values
            spline = RegularGridInterpolator(
                (x_coordinates, y_coordinates),
                z_values,
                method=method,
            )
            self.spline_memo[args] = spline
            return spline

    def create_3Dspline(
        self,
        variable,
        xdimension,
        ydimension,
        zdimension,
        coords={},
        method="linear",
    ):
        """Memoized function that interpolates splines for any variable along
        two arbitrary axes with scipy.interpolate.RegularGridInterpolator.
        Default interpolation method is linear.

        Args:
            variable (str): index the specific DataArray to fit the spline to
            xdimension (str): dimension name of the x-coordinate
            ydimension (str): dimension name of the y-coordinate
            zdimensoin (str): dimension name of the z-coordinate
            coords (dict): a dictionary of the form {dimension: value} for all
            other dimensions apart from the x and y coordinate dimensions. See
            create_1Dspline for a similar method.

        Returns:
            Callable: An RegularGridInterpolator function.
        """
        coords_hashable = self._dict_to_hashable(coords)
        args = (variable, xdimension, ydimension, coords_hashable)
        if args in self.spline_memo.keys():
            return self.spline_memo[args]
        else:
            data = self.dataset[variable]
            x_coordinates = self.get_coordinate_array(xdimension)
            y_coordinates = self.get_coordinate_array(ydimension)
            z_coordinates = self.get_coordinate_array(zdimension)
            func_values = (
                data.loc[coords].transpose(xdimension, ydimension, zdimension).values
            )
            spline = RegularGridInterpolator(
                points=(x_coordinates, y_coordinates, z_coordinates),
                values=func_values,
                method=method,
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

        coords = {
            "frequency": freq_coord,
            "toroidal_angle": tor_coord,
            "poloidal_angle": pol_coord,
        }

        reflection_mask = self.generate_reflection_mask()
        return reflection_mask[coords]

    def check_cutoff_indices(self):
        """Checks for and interpolates problematic cutoff indices (negative integers that
        occur when casting np.NaN to int).
        """
        cutoff_indices = self.dataset["cutoff_index"]
        # cast to float to work with np.NaN values
        float_indices = cutoff_indices.astype(float)
        index_list = np.argwhere(float_indices.values < 0)
        print("Checking 'cutoff_index'...")
        print(f"Number of problematic indices: {len(index_list)}")
        if not len(index_list):
            return []
        interp_indices = float_indices.where(float_indices >= 0, other=np.nan)

        for dimension in ("toroidal_angle", "poloidal_angle", "frequency"):
            print(f"trying {dimension}")
            try:
                interp_indices = interp_indices.interpolate_na(
                    dim=dimension,
                    method="nearest",
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

    def set_problematic_indices(self, value=-1):
        """Bandaid fix for stubborn missing indices that cannot be fixed by interpolation."""
        cutoff_indices = self.dataset["cutoff_index"]
        fixed_indices = cutoff_indices.where(cutoff_indices >= 0, other=value)
        self.dataset["cutoff_index"] = fixed_indices

        new_list = np.argwhere(fixed_indices.values < 0)
        if len(new_list):
            print(
                f"Failed to resolve problematic indices. Consider masking the affected regions. Failed indices: {new_list}"
            )
        return xr.where(cutoff_indices < 0, 1, 0)

    def check_float_arrays(self, variable, method="linear"):
        """Checks for and interpolates any null array values."""
        new_array = self.dataset[variable]
        print(f"Checking {variable}...")
        nan_entries = len(np.argwhere(new_array.isnull().values))
        print("Number of NaN entries:", nan_entries)
        if not nan_entries:
            return []
        # Try multiple dimensions as interpolating in one axis may not eliminate all gaps
        for dimension in ("toroidal_angle", "poloidal_angle", "frequency"):
            print(f"trying {dimension}")
            try:
                new_array = new_array.interpolate_na(
                    dim=dimension,
                    method=method,
                )
            except Exception:
                print(
                    f"{variable} interpolation along {dimension} failed. Trying new axis..."
                )
                continue

        index_list = np.argwhere(new_array.isnull().values)

        if len(index_list):
            print(
                f"Failed to resolve problematic {variable} values. Failed indices: {index_list}. Starting ND interpolation.."
            )

            valid_indices = np.logical_not(np.isnan(new_array.values))

            return index_list

        else:
            self.dataset[variable] = new_array

    def check_all_float_arrays(self):
        """Calls check_float_arrays for all relevant variables."""
        for variable in self.variables:
            if variable != "cutoff_index":
                self.check_float_arrays(variable)

    def _dict_to_hashable(self, dictionary):
        """Internal method for converting dictionary arguments to a hashable form
        for memoization.
        """
        return tuple(dictionary.items())

    def _hashable_to_dict(self, hashable):
        return dict(hashable)

    ### Plotting Methods ###

    def imshow_slice(
        self,
        variable,
        xdimension,
        ydimension,
        coords={},
        cmap="plasma_r",
        **kwargs,
    ):
        """Visualizes a specified slice of the DataArray with plt.imshow.

        Args:
            variable(str): DataArray variable to visualize
            xdimension (str): Dimension for the x-axis
            ydimension (str): Dimension for the y-axis
            coords (dict): Dictionary of coordinate values of the other
            dimensions to be held constant
            cmap (str): Colormap to be used
            kwargs (optional): Optional keyword arguments to pass to plt.imshow
        """
        data_slice = self.dataset[variable].loc[coords]
        if len(data_slice.shape) != 2:
            raise ValueError("Inappropriate number of constant dimensions supplied")
        title_string = ""
        for key, value in coords.items():
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
            data_slice,
            cmap=cmap,
            origin="lower",
            extent=extent,
            aspect="auto",
            **kwargs,
        )
        plt.colorbar(mappable=im)
        return fig, ax

    def plot_contour(
        self,
        variable,
        xdimension,
        ydimension,
        coords={},
        cmap="plasma_r",
        mask_flag=False,
        levels=20,
        **kwargs,
    ):
        """Make a contour plot of a specific slice of the DataArray.

        Args:
            variable(str): DataArray variable to visualize
            xdimension (str): Dimension for the x-axis
            ydimension (str): Dimension for the y-axis
            coords (dict): Dictionary of coordinate values of the other
            dimensions to be held constant
            cmap (str): Colormap to be used
            mask_flag(bool): Whether or not to mask non-reflected rays. Only available
            for dimensions 'frequency', 'poloidal_angle', 'toroidal_angle'.

        Returns:
            Artist objects fig, ax
        """
        data_spline = self.create_2Dspline(variable, xdimension, ydimension, coords)

        title_string = ""
        for key, value in coords.items():
            title_string += f",{key}={value} "
        x_coords = self.get_coordinate_array(xdimension)
        y_coords = self.get_coordinate_array(ydimension)
        x = np.linspace(x_coords[0], x_coords[-1], 500)
        y = np.linspace(y_coords[0], y_coords[-1], 500)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

        plot_array = data_spline((X, Y))

        fig, ax = plt.subplots()
        if mask_flag:
            mask_coords = coords.copy()
            if "trajectory_step" in mask_coords.keys():
                mask_coords.pop("trajectory_step")
            reflection_mask = (
                self.dataset["reflection_mask"]
                .loc[mask_coords]
                .transpose(xdimension, ydimension)
                .values
            )
            mask_function = RegularGridInterpolator(
                (x_coords, y_coords),
                reflection_mask,
                method="nearest",
                bounds_error=False,
            )
            interp_mask = mask_function((X, Y))
            plot_masked = np.ma.array(plot_array, mask=interp_mask)

            CS = ax.contourf(
                X,
                Y,
                plot_masked,
                levels=levels,
                cmap="plasma_r",
                corner_mask=True,
                **kwargs,
            )
            CSgo = ax.contour(
                X,
                Y,
                plot_masked,
                levels=levels,
                colors="k",
                corner_mask=True,
                **kwargs,
            )

        else:
            CS = ax.contourf(
                X,
                Y,
                plot_array,
                levels=levels,
                cmap="plasma_r",
                corner_mask=True,
                **kwargs,
            )
            CSgo = ax.contour(
                X,
                Y,
                plot_array,
                levels=levels,
                colors="k",
                corner_mask=True,
                **kwargs,
            )
        fig.suptitle(f"{variable}, " + title_string)
        ax.set_xlabel(f"{xdimension}")
        ax.set_ylabel(f"{ydimension}")
        ax.clabel(CSgo, inline=True, fontsize=6)
        fig.colorbar(CS)
        return fig, ax

    def plot_cutoff_contour(
        self,
        coords,
        save_path=None,
        measure="rho",
        mask_flag=True,
        bounds=None,
    ):
        """Creates a contour plot of toroidal/poloidal steering vs. frequency, with
        contours of constant cutoff (either rho or metres). A bivariate spline is fitted
        to the data to produce the plot.

        Args:
            coords (dict): Dictionary key must be either 'toroidal_angle' or 'poloidal_angle',
            with the value provided being the angle held constant for.
            measure (str, optional): Specify 'rho' or 'm' to plot in terms of normalized
            poloidal radius or actual poloidal distance. Default is 'rho'.
            mask_flag (bool, optional): Select whether to mask the data with 'reflection_mask'.
            Default is True.
            bounds (tup, optional): Specify plotting range in the form (x_start, x_stop,
            y_start, y_stop)

        Returns:
            Artist objects fig, ax
        """
        const_angle_str = list(coords.keys())[0]
        const_angle = list(coords.values())[0]
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
            variable = "cutoff_rho"
        elif measure == "m":
            title_label = "dist"
            variable = "cutoff_distance"
        else:
            raise ValueError("Provided measure is not 'rho' or 'm'.")

        if const_angle not in self.get_coordinate_array(const_angle_str):
            raise ValueError(
                f"Value of {const_angle_str} not in original coordinate array."
            )

        coords = {const_angle_str: const_angle}
        xdimension = "frequency"
        ydimension = var_angle_str
        spline_surface = self.create_2Dspline(variable, xdimension, ydimension, coords)

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

        # Interpolate and apply the reflection mask to remove inaccessible regions
        reflection_mask = (
            self.dataset["reflection_mask"]
            .loc[coords]
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
                Xgrid,
                Ygrid,
                Z_masked,
                levels=20,
                colors="w",
                corner_mask=True,
            )
        else:
            CS = ax.contourf(
                Xgrid, Ygrid, Zgrid, levels=20, cmap="plasma_r", corner_mask=True
            )
            CSgo = ax.contour(
                Xgrid,
                Ygrid,
                Z_grid,
                levels=20,
                colors="w",
                corner_mask=True,
            )

        fig.suptitle(f"Cutoff {title_label}, {const_angle_str}={const_angle}$^\circ$")
        ax.set_xlabel("frequency/GHz")
        ax.set_ylabel(f"{var_angle_str}/$^\circ$")
        ax.clabel(CSgo, inline=True, fontsize=6)
        fig.colorbar(CS)
        return fig, ax

    def plot_cutoff_positions(
        self,
        toroidal_angle=0,
        show_pitch=False,
        xlim=None,
        ylim=None,
        levels=20,
        flux_cmap="plasma_r",
        position_cmap="spectral",
        pitch_cmap="plasma_r",
        pitch_max=None,
        pitch_min=None,
    ):
        """Plots the cutoff locations of all ray trajectories for a specified toroidal steering
        in the poloidal (R-Z) plane. Can be plotted against a colour contour map of the pitch
        angles in the R-Z plane.

        Args:
            toroidal_angle (int/float, optional): Toroidal angle to plot the cutoff positions
            Default 0.
            show_pitch (bool, optional): Controls whether or not to show pitch angle contours
            Default False.
            xlim (tuple of floats, optional): Tuple passed to xlim argument of matplotlib.contour
            ylim (tuple of floats, optional): Tuple passed to ylim argument of matplotlib.contour
            levels (int, optional): Number of contour levels to plot
            flux_cmap (str, optional): Colormap assigned to the flux contours. Ignored and set to
            black if show_pitch is True. Default "plasma_r".
            position_cmap (str, optional): Colormap assigned to the position markers by poloidal
            steering. Default "spectral".
            pitch_cmap (str, optional): Colormap assigned to the pitch angle contours. Default
            "plasma_r".
            pitch_max (float, optional): Passed to the vmax argument when plotting pitch angle
            contours.
            pitch_min (float, optional): Passed to the vmin argument when plotting pitch angle
            contours.

        Returns:
            Artist objects fig, ax
        """

        fig, ax = plt.subplots()
        rho = np.sqrt(self.topfile["pol_flux"].transpose("Z", "R"))

        if show_pitch:
            pitch = self.topfile["pitch_angle"].transpose("Z", "R")
            cont = rho.plot.contour(
                levels=levels,
                vmax=1.0,
                xlim=xlim,
                ylim=ylim,
                colors="k",
            )
            pitch.plot.contourf(
                levels=levels,
                xlim=xlim,
                ylim=ylim,
                cmap=pitch_cmap,
                vmax=pitch_max,
                vmin=pitch_min,
                add_colorbar=True,
            )
        else:
            cont = rho.plot.contour(
                levels=levels,
                vmax=1.0,
                xlim=xlim,
                ylim=ylim,
                cmap=flux_cmap,
            )

        ax.clabel(cont, inline=True, fontsize=5)
        mask = self.dataset["reflection_mask"]
        cutoff_R = self.dataset["cutoff_q_R_array"].where(np.logical_not(mask))
        cutoff_Z = self.dataset["cutoff_q_Z_array"].where(np.logical_not(mask))
        poloidal_angles = self.get_coordinate_array("poloidal_angle")
        frequencies = self.get_coordinate_array("frequency")
        for frequency in frequencies:
            R = cutoff_R.loc[{"frequency": frequency, "toroidal_angle": toroidal_angle}]
            Z = cutoff_Z.loc[{"frequency": frequency, "toroidal_angle": toroidal_angle}]
            pol_array = (R.coords["poloidal_angle"].values + 15) / 30
            sc = ax.scatter(
                R,
                Z,
                s=5,
                c=pol_array,
                cmap=position_cmap,
                edgecolors="k",
                linewidths=0.5,
                marker="o",
            )
        norm = mpl.colors.Normalize(vmin=-15, vmax=15)
        plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=position_cmap),
            label="Poloidal Angle/$^\circ$",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("R/m")
        ax.set_ylabel("Z/m")
        fig.suptitle(f"Cutoff positions, tor={toroidal_angle}$^\circ$", fontsize=15)
        return fig, ax

    def plot_opt_tor_contour(self, opt_tor_type="mismatch"):
        """
        Plots contours of the optimal toroidal steering angle in the frequency-poloidal angle
        parameter space.

        Return:
            Artist objects fig, ax
        """
        freq = self.get_coordinate_array("frequency")
        pol = self.get_coordinate_array("poloidal_angle")
        x = np.linspace(freq[0], freq[-1], 500)
        y = np.linspace(pol[0], pol[-1], 500)
        X, Y = np.meshgrid(x, y, indexing="ij")
        spline = self.create_2Dspline(
            variable=f"opt_tor_{opt_tor_type}",
            xdimension="frequency",
            ydimension="poloidal_angle",
        )
        data = spline((X, Y))
        mask_func = RegularGridInterpolator(
            points=(freq, pol),
            values=self.dataset[f"opt_tor_{opt_tor_type}_mask"]
            .transpose("frequency", "poloidal_angle")
            .values,
            method="nearest",
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

    def plot_mismatch_angles(self, tor_upper, tor_lower, frequency, poloidal_angle):
        """Plots the mismatch angle along with an interpolated spline for the specified
        frequency, poloidal angle, and toroidal steering range.

        Args:
            tor_upper (float): Upper range of toroidal steerings.
            tor_lower (float): Lower range of toroidal steerings.
            frequency (float): Frequency.
            poloidal_angle (float): Poloidal Angle.
        """
        theta_m = (
            self.dataset["cutoff_theta_m"]
            .loc[{"frequency": frequency, "poloidal_angle": poloidal_angle}]
            .sel(toroidal_angle=slice(tor_lower, tor_upper))
        )
        toroidal_angles = theta_m.coords["toroidal_angle"].values
        spline = UnivariateSpline(toroidal_angles, theta_m.values)
        toroidal_range = np.linspace(tor_lower, tor_upper, 200)
        fig, ax = plt.subplots()
        ax.scatter(toroidal_angles, theta_m, marker="o", label="points")
        ax.plot(toroidal_range, spline(toroidal_range), label="spline fit")
        fig.suptitle("Mismatch angle vs. toroidal angle", fontsize=15)
        return fig, ax


## TODO: Expand this as a subclass of SweepDataset, but calling the analysis method
## iteratively calls the methods for all the Datasets within it.
## Alternatively can read the xarrays from the SweepDatasets and concatenate them
## into an even larger xarray with current scaling or some other parameter as the
## Concatenated dimension? Then extend with base class + subclasses.


class MultiSweeps:
    """Class for holding and analysing data from multiple equilibrium sweeps stored
    as a SweepDataset class. Initializied by providing an arbitrary number of
    SweepDataset objects with key values corresponding to the descriptor attribute
    of SweepDataset objects.

    Note that the analysis methods associated with this class assumes that the sweeps
    have the same initial beam properties and simulation parameters, with the only
    difference being the equilibrium properties of the plasma.
    """

    def __init__(
        self,
        frequencies,
        poloidal_angles,
        toroidal_angles,
        datasets,
    ):
        """Initializes a MultiSweep class.

        Args:
            frequencies (array): The common 'frequency' coordinate array between all
            SweepDatasets loaded.
            poloidal_angles (array): The common 'poloidal_angle' coordinate array between all
            SweepDatasets loaded.
            toroidal_angles (array): The common 'toroidal_angle' coordinate array between all
            SweepDatasets loaded.
            datasets (iterable): An iterable of SweepDataset objects to load into the class.
            All SweepDataset objects must have a descriptor attribute set.
        """
        # Common coordinate arrays of the SweepDatasets
        self.frequencies = frequencies
        self.poloidal_angles = poloidal_angles
        self.toroidal_angles = toroidal_angles

        self.datasets = {}
        for dataset in datasets:
            self.datasets[dataset.descriptor] = dataset

        self.tags = {
            "hi": None,
            "lo": None,
            "1.0x": None,
        }

    def add_SweepDataset(self, SweepDataset):
        self.datasets[SweepDataset.descriptor] = SweepDataset

    def remove_SweepDataset(self, descriptor):
        item = self.datasets[descriptor]
        del self.datasets[descriptor]
        return item

    def get_descriptors(self):
        """
        Returns:
            List: A list of descriptors corresponding to the SweepDatasets loaded in
            this class.
        """
        return list(self.datasets.keys())

    def plot_opt_tors(self, poloidal_angles=None, opt_tor_type="mismatch"):
        """Plots the optimal toroidal steering vs. frequency for a set of poloidal angles,
        for all of the SweepDatasets loaded onto this MultiSweep instance.

        Args:
            poloidal_angles (array): An array of poloidal angles to plot the optimal toroidal
            steerings for. Default is the poloidal_angle coordinate array assigned when
            initializing the class.

        Returns:
            Artist objects fig, ax
        """

        if poloidal_angles == None:
            poloidal_angles = self.poloidal_angles

        freq_range = np.linspace(self.frequencies[0], self.frequencies[-1], 500)

        linestyles = ["solid", "dashed", "dashdot", "dotted"]
        colormap = plt.cm.gnuplot2
        markers = split_list(mpl.lines.Line2D.filled_markers)
        pol_count = len(poloidal_angles)
        counter = 0
        descriptor_list = []

        fig, ax = plt.subplots()

        for poloidal_angle in poloidal_angles:
            color = colormap(counter / pol_count)
            counter2 = 0
            for dataset in self.datasets.values():
                if counter == 0:
                    descriptor_list.append(dataset.descriptor)
                marker = markers[counter2]
                opt_tor_spline = dataset.create_1Dspline(
                    variable=f"opt_tor_{opt_tor_type}",
                    dimension="frequency",
                    coords={"poloidal_angle": poloidal_angle},
                )
                opt_tor_array = opt_tor_spline(freq_range)
                ax.plot(
                    freq_range,
                    opt_tor_array,
                    color=color,
                    alpha=0.5,
                )
                ax.scatter(
                    self.frequencies[::2],
                    opt_tor_spline(self.frequencies[::2]),
                    marker=marker,
                    s=10,
                    color=color,
                )
                counter2 += 1
            counter += 1

        Line2D = mpl.lines.Line2D
        custom_colors = [
            Line2D([0], [0], color=colormap(count / pol_count), lw=4)
            for count in range(pol_count)
        ]
        custom_markers = [
            Line2D([0], [0], color="k", marker=markers[count])
            for count in range(pol_count)
        ]
        custom_handles = custom_colors + custom_markers
        custom_labels = [
            f"pol={pol_angle}" for pol_angle in poloidal_angles
        ] + descriptor_list
        fig.suptitle("Range of optimal steerings vs. scaling", fontsize=15)
        ax.legend(
            custom_handles,
            custom_labels,
            bbox_to_anchor=(1.05, 0),
            loc="lower left",
            borderaxespad=0.0,
            fontsize=8,
        )
        ax.set_xlabel("Frequency/GHz")
        ax.set_ylabel("Opt. toroidal steering/$^\circ$")

        return fig, ax

    def plot_rho_freq_relations(self, poloidal_angle, toroidal_angle=0):
        """Plots multiple overlapping cutoff rho vs. frequency plots for
        each SweepDataset loaded onto the MultiSweep class.

        Returns:
            Artist object fig
        """

        freq_range = np.linspace(self.frequencies[0], self.frequencies[-1], 500)
        colormap = plt.cm.gnuplot2
        counter = 0
        total = len(self.datasets.keys())
        fig = plt.figure()
        for dataset in self.datasets.values():
            color = colormap(counter / total)
            spline = dataset.get_rho_freq_spline(poloidal_angle, toroidal_angle)
            plt.plot(
                freq_range,
                spline(freq_range),
                color=color,
                alpha=0.5,
                label=f"{dataset.descriptor}",
            )
            counter += 1

        plt.title("Cutoff $\\rho$ vs. Frequency")
        plt.xlabel("Frequency/GHz")
        plt.ylabel("$\\rho$")
        plt.legend()
        return fig


class TrajectoryPlot:
    """Convenience class for plotting ray trajectories on top of a set
    of poloidal flux contours loaded from a json topfile.
    """

    def __init__(self, topfile_path):
        with open(topfile_path) as f:
            topfile = json.load(f)
            R_coord = topfile["R"]
            Z_coord = topfile["Z"]
            polflux = topfile["pol_flux"]
            self.polflux = xr.DataArray(
                data=np.transpose(polflux),
                dims=("R", "Z"),
                coords={
                    "R": R_coord,
                    "Z": Z_coord,
                },
            )

    def plot_pol_trajectory(self, q_R, q_Z, xlim=(1.25, 2.5), ylim=(-1.2, 1.2)):
        fig = plt.figure()
        R_coord = self.polflux.coords["R"].values
        Z_coord = self.polflux.coords["Z"].values
        data = np.clip(self.polflux.transpose("Z", "R"), 0, 1)
        xr.plot.contourf(
            darray=data,
            levels=50,
            xlim=xlim,
            ylim=ylim,
            add_colorbar=True,
            cmap="plasma_r",
        )
        xr.plot.contour(
            darray=data,
            levels=20,
            xlim=xlim,
            ylim=ylim,
            colors="w",
        )
        plt.plot(q_R, q_Z, label="trajectory", color="k")
        plt.legend()
        return fig


## Helper functions
def split_list(a_list):
    i_half = len(a_list) // 2
    return a_list[:i_half] + a_list[i_half:]


def fit_gaussian(toroidal_angle, opt_tor, width):
    return np.exp(-2 * (((toroidal_angle - opt_tor) / width) ** 2))


def gaussian(theta_m, delta):
    return np.exp(-2 * ((theta_m / delta) ** 2))


def noisy_gaussian(theta_m, delta, std=0.05):
    mean = gaussian(theta_m, delta)
    return mean + np.random.normal(0, std * mean)


def find_nearest(array, value):
    index = np.abs((array - value)).argmin()
    return array[index]


def scale_array(array):
    maxval = array.max()
    return array / maxval
