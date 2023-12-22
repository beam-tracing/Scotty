"""Class to simulate pitch diagnostic design configurations.
Complements sweep_analysis for analysing sweeps with varying
poloidal steering and frequencies.

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
    PPoly,
    splrep,
)
from scipy.optimize import newton
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib as mpl
from scotty.fun_general import find_q_lab_Cartesian
from itertools import product
from scotty.sweep_analysis import (
    SweepDataset,
    fit_gaussian,
    gaussian,
    noisy_gaussian,
    find_nearest,
    scale_array,
)


class PitchDiagnostic:
    """Separate class for representing a particular pitch angle diagnostic design/setup.
    Analysis methods take in Scotty simulation data in a SweepDataset class, and saves the
    output in a dictionary of datasets. Each dataset corresponds to a different SweepDataset
    equilibriums. Datasets are organized into NetCDF groups when reading and writing to file.

    Class is constructed by calling class methods create_new or from_netcdf, and has the
    following attributes:
        self.home: Stores information associated with the diagnostic configuration.
        self.ds_dict: A dictionary of Datasets, each read from a SweepDataset instance.
        self.topfiles: A dictionary of Datasets, each stores information from a json
        topfile and is associated with a SweepDataset instance.
        self.pitch_ds: A dataset to store pitch angle information analyzed from the
        set of topfiles loaded to this class.

    """

    def __init__(self, ds):
        self.home = ds
        self.ds_dict = {}
        self.topfiles = {}
        self.pitch_ds = None

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
        """Creates a PitchDiagnostic instance by reading PitchDiagnostic data previously saved to
        a netcdf file format.

        Args:
            path_string (str): Path string of the .nc file.
        """
        with xr.open_dataset(path_string, group="home", engine="netcdf4") as ds:
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

    def from_topfile_path(self, topfile_path, descriptor):
        """Reads data from a json topfile and loads it onto an xarray Dataset
        saved to the self.topfiles dictionary.

        Args:
            topfile_path (str): Path string to the json topfile.
            descriptor (str): Descriptor string of the SweepDataset instance that
            the topfile is associated with.
        """
        self.topfiles[descriptor] = xr.Dataset()
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
            self.topfiles[descriptor][var_name] = xr.DataArray(
                data=array,
                dims=("R", "Z"),
                coords={
                    "R": R_coord,
                    "Z": Z_coord,
                },
            )

        ds = self.topfiles[descriptor]
        # Find pitch angle
        ds["pitch_angle"] = np.rad2deg(
            np.arctan(
                np.divide(
                    np.hypot(ds["Br"], ds["Bz"]),
                    ds["Bt"],
                )
            )
        )

        r_index = np.squeeze(ds["pol_flux"].loc[{"Z": 0}].argmin().values)
        magnetic_axis = np.array((ds.coords["R"][r_index], 0))
        z_index = np.squeeze(np.argwhere(ds.coords["Z"].values == 0))
        ds.attrs["magnetic_axis"] = magnetic_axis
        ds.attrs["axis_index"] = np.array((r_index, z_index))

    def read_topfiles(self):
        """Calls from_topfile_path on all currently assigned topfile path strings."""
        for descriptor in self.descriptors:
            try:
                path = self.get_topfile_path(descriptor)
                self.from_topfile_path(path, descriptor)
                print(f"Data read from {descriptor} topfile.")
            except ValueError as error:
                print(error)
                continue

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

    @property
    def descriptors(self):
        out = []
        for key in self.ds_dict.keys():
            if key != "cutoff_data":
                out.append(key)
        return out

    ## set/get methods
    def reorder_keys(self, descriptor_tuple):
        """Sets the ordering of the datasets assigned to each descriptor.

        Args:
            descriptor_tuple (tuple): An ordered tuple of descriptors to reorder
            the existing set of Datasets in self.ds_dict.
        """
        new_dict = {}
        for descriptor in descriptor_tuple:
            new_dict[descriptor] = self.ds_dict.pop(descriptor)
        self.ds_dict = new_dict
        return tuple(self.ds_dict.keys())

    def set_current_scaling_attr(self, descriptor, value):
        """Sets the current scaling attribute used in some plots.

        Args:
            descriptor (str): Descriptor string of Dataset to set the attribute.
            value (str): Value of the current scaling attribute.
        """
        self.ds_dict[descriptor].attrs["current_scaling"] = value

    def get_keys(self):
        """
        Returns:
            keys_dict (dictionary): A dictionary consisting of descriptors
            and a tuple of the variables in the corresponding Dataset.
        """
        keys_dict = {}
        for key in self.ds_dict.keys():
            values = self.ds_dict[key].keys()
            keys_dict[key] = tuple(values)
        print(keys)
        return keys_dict

    def set_std_noise(self, new_value):
        """Set a new value for the standard gaussian noise associated with
        this PitchDiagnostic configuration.

        Args:
            new_value (float): A number from 0 to 1 representing the fractional
            standard deviation.
        """
        self.home.attrs["std_noise"] = new_value

    def set_rho_freq_relation(self, SweepDataset):
        """Sets the default rho-freq relation to use to map frequency to spatial
        rho coordinates.

        Args:
            SweepDataset (SweepDataset): The SweepDataset object representing the
            default equilibrium to be assumed.
        """
        if "cutoff_data" in self.ds_dict.keys():
            rho_freq_spline = self.get_rho_freq_spline()
            return rho_freq_spline

        frequency_range = np.linspace(self.frequencies[0], self.frequencies[-1], 500)
        spline = SweepDataset.create_3Dspline(
            variable="cutoff_rho",
            xdimension="frequency",
            ydimension="poloidal_angle",
            zdimension="toroidal_angle",
            method="pchip",
        )
        X, Y, Z = np.meshgrid(
            frequency_range, self.poloidal_angles, self.toroidal_angles, indexing="ij"
        )
        cutoff_rho = spline((X, Y, Z))
        ds = xr.Dataset()
        ds["cutoff_rho"] = xr.DataArray(
            data=cutoff_rho,
            dims=("frequency_range", "poloidal_angle", "toroidal_angle"),
            coords={
                "frequency_range": frequency_range,
                "poloidal_angle": self.poloidal_angles,
                "toroidal_angle": self.toroidal_angles,
            },
        )
        self.ds_dict["cutoff_data"] = ds
        rho_freq_relation = ds["cutoff_rho"].mean(dim="toroidal_angle").squeeze()
        self.ds_dict["cutoff_data"]["rho_freq_relation"] = rho_freq_relation
        rho_freq_spline = self.get_rho_freq_spline()

        ## Calculate upper and lower ranges for rho for a given frequency
        rho_upper = ds["cutoff_rho"].max(dim="toroidal_angle").squeeze()
        rho_lower = ds["cutoff_rho"].min(dim="toroidal_angle").squeeze()
        self.ds_dict["cutoff_data"]["rho_upper"] = rho_upper
        self.ds_dict["cutoff_data"]["rho_lower"] = rho_lower

        return rho_freq_spline

    def get_rho_freq_spline(self, order=5, smoothing=0):
        """Spline for rho as a function of frequency.
        Args:
            order (int): Passed to 'k' argument of UnivariateSpline
            smoothing (int): Passed to 's' argument of UnivariateSpline
        Returns:
            UnivariateSpline
        """
        frequency_range = np.linspace(self.frequencies[0], self.frequencies[-1], 500)
        return UnivariateSpline(
            frequency_range,
            self.ds_dict["cutoff_data"]["rho_freq_relation"],
            k=order,
            s=smoothing,
            ext=3,
        )

    def get_freq_rho_spline(self, order=5, smoothing=0):
        """Spline for frequency as a function of rho.
        Args:
            order (int): Passed to 'k' argument of UnivariateSpline
            smoothing (int): Passed to 's' argument of UnivariateSpline
        Returns:
            UnivariateSpline
        """
        frequency_range_r = np.linspace(self.frequencies[-1], self.frequencies[0], 500)
        return UnivariateSpline(
            self.ds_dict["cutoff_data"]["rho_freq_relation"][::-1],
            frequency_range_r,
            k=order,
            s=smoothing,
            ext=3,
        )

    def get_rho_delta_splines(self, order=5, smoothing=0):
        """Generates splines for the upper and lower bounds of rho as a function
        of frequency across the toroidal range.
        Args:
            order (int): Passed to 'k' argument of UnivariateSpline
            smoothing (int): Passed to 's' argument of UnivariateSpline
        Returns:
            UnivariateSpline, UnivariateSpline: the first interpolates the upper
            limit while the second interpolates the lower limit
        """
        frequency_range = np.linspace(self.frequencies[0], self.frequencies[-1], 500)
        upper_spline = UnivariateSpline(
            frequency_range,
            self.ds_dict["cutoff_data"]["rho_upper"],
            k=order,
            s=smoothing,
        )
        lower_spline = UnivariateSpline(
            frequency_range,
            self.ds_dict["cutoff_data"]["rho_lower"],
            k=order,
            s=smoothing,
        )
        return upper_spline, lower_spline

    def get_midplane_pitch_rho_spline(self, descriptor, order=5, smoothing=0):
        """Spline for pitch angle as a function of rho on the midplane.
        Args:
            descriptor (int): Descriptor of the Dataset to generate the spline for.
            order (int): Passed to 'k' argument of UnivariateSpline
            smoothing (int): Passed to 's' argument of UnivariateSpline
        Returns:
            UnivariateSpline
        """
        ds = self.topfiles[descriptor]
        if "midplane_pitch_rho_profile" in ds.keys():
            da = ds["pitch_rho_profile"]
            rho_range = da.coords["midplane_rho"]
            return UnivariateSpline(
                x=rho_range, y=da.values, k=order, s=smoothing, ext=3
            )

        r_index, z_index = ds.attrs["axis_index"]
        midplane_rho = np.sqrt(ds["pol_flux"].loc[{"Z": 0}][r_index:])
        midplane_pitch = ds["pitch_angle"].loc[{"Z": 0}][r_index:]
        pitch_rho = UnivariateSpline(
            x=midplane_rho, y=midplane_pitch, k=order, s=smoothing, ext=3
        )
        rho_range = np.linspace(midplane_rho[0], midplane_rho[-1], 500)
        ds["midplane_pitch_rho_profile"] = xr.DataArray(
            data=pitch_rho(rho_range),
            dims=("midplane_rho"),
            coords={"midplane_rho": rho_range},
        )
        return pitch_rho

    def get_pitch_rho_spline(self, descriptor, order=5, smoothing=0):
        """Spline for pitch angle as a function of rho at the cutoff positions.
        Args:
            descriptor (int): Descriptor of the Dataset to generate the spline for.
            order (int): Passed to 'k' argument of UnivariateSpline
            smoothing (int): Passed to 's' argument of UnivariateSpline
        Returns:
            UnivariateSpline
        """
        topfile = self.topfiles[descriptor]
        ds = self.ds_dict[descriptor]
        if "pitch_rho_profile" in ds.keys():
            da = ds["pitch_rho_profile"]
            rho_range = da.coords["cutoff_rho"]
            return UnivariateSpline(x=rho_range[::-1], y=da.values[::-1], s=0, ext=3)

        cutoff_r = ds["cutoff_R"]
        cutoff_z = ds["cutoff_Z"]
        cutoff_pitch = topfile["pitch_angle"].sel(
            R=cutoff_r,
            Z=cutoff_z,
            method="nearest",
        )
        cutoff_rho = np.sqrt(
            topfile["pol_flux"].sel(R=cutoff_r, Z=cutoff_z, method="nearest").squeeze()
        )
        pitch_rho = UnivariateSpline(
            x=cutoff_rho[::-1], y=cutoff_pitch[::-1], k=order, s=smoothing, ext=3
        )
        rho_range = np.linspace(cutoff_rho[0], cutoff_rho[-1], 500)
        ds["pitch_rho_profile"] = xr.DataArray(
            data=pitch_rho(rho_range),
            dims=("cutoff_rho"),
            coords={"cutoff_rho": rho_range},
        )
        return pitch_rho

    def get_opt_tor_freq_spline(
        self, descriptor, order=5, smoothing=0, opt_tor_type="mismatch"
    ):
        """Spline for optimal toroidal steering as a function of launch frequency.
        Args:
            descriptor (int): Descriptor of the Dataset to generate the spline for.
            order (int): Passed to 'k' argument of UnivariateSpline
            smoothing (int): Passed to 's' argument of UnivariateSpline
        Returns:
            UnivariateSpline
        """
        ds = self.ds_dict[descriptor]
        opt_tor_freq = ds[f"opt_tor_{opt_tor_type}_profile"]
        return UnivariateSpline(
            x=np.linspace(self.frequencies[0], self.frequencies[-1], 500),
            y=opt_tor_freq,
            k=order,
            s=smoothing,
            ext=3,
        )

    def get_opt_tor_rho_spline(self, descriptor, order=5, smoothing=0):
        """Spline for optimal toroidal steering as a function of cutoff rho.
        Args:
            descriptor (int): Descriptor of the Dataset to generate the spline for.
            order (int): Passed to 'k' argument of UnivariateSpline
            smoothing (int): Passed to 's' argument of UnivariateSpline
        Returns:
            UnivariateSpline
        """
        opt_tor_freq = self.get_opt_tor_freq_spline(
            descriptor, order=order, smoothing=smoothing
        )
        freq_rho = self.get_freq_rho_spline(order=order, smoothing=smoothing)
        return lambda x: opt_tor_freq(freq_rho(x))

    # For getting pitch angle profiles from corresponding topfiles

    def set_topfile_path(self, topfile_path, descriptor):
        """Sets a path string to a json topfile as an attributed of the associated
        Dataset.

        Args:
            topfile_path (str): Path string to the json topfile
            descriptor (str): Descriptor string of the corresponding SweepDataset
        """
        if descriptor not in self.descriptors:
            raise ValueError(f"{descriptor} not saved to PitchDiagnostic instance.")
        self.ds_dict[descriptor].attrs["topfile_path"] = topfile_path

    def get_topfile_path(self, descriptor):
        if "topfile_path" in self.ds_dict[descriptor].attrs:
            return self.ds_dict[descriptor].attrs["topfile_path"]
        raise ValueError(f"No topfile path associated with {descriptor}")

    def set_topfile_paths(self, path_dict):
        """Calls set_topfile_path on all key-value pairs in the provided dictionary
        corresponding to the descriptor and topfile_path respectively.

        Args:
            path_dict (dict): A dictionary with descriptor keys and path strings to the
            corresponding json topfile.
        """
        if not set(path_dict.keys()).issubset(self.descriptors):
            raise ValueError("Invalid descriptor key provided!")
        for key, value in path_dict.items():
            self.set_topfile_path(value, key)

    ## Simulation methods

    def get_cutoff_locations(self, SweepDataset, opt_tor_type="mismatch"):
        # opt_tor_type is used as a class-wide toggle since many plots rely on the
        # cutoff locations calculated; kind of a stop-gap solution
        """Reads the cutoff locations in R-Z coordinates from the SweepDataset.
        Only data corresponding to the poloidal steering, toroidal range and
        frequencies of the PitchDiagnostic is read and stored.

        Args:
            SweepDataset (SweepDataset): SweepDataset to read from.
        """
        descriptor = str(SweepDataset.descriptor)
        if descriptor in self.ds_dict:
            ds = self.ds_dict[descriptor]
            """
            if {"cutoff_R", "cutoff_Z"}.issubset(set(ds.keys())):
                print(f"Cutoff positions already read for {descriptor}.")
                return None"""
        else:
            self.ds_dict[descriptor] = xr.Dataset()

        cutoff_R_spline = SweepDataset.create_2Dspline(
            variable="cutoff_q_R_array",
            xdimension="frequency",
            ydimension="toroidal_angle",
            method="pchip",
            coords={"poloidal_angle": self.poloidal_angles[0]},
        )

        cutoff_Z_spline = SweepDataset.create_2Dspline(
            variable="cutoff_q_Z_array",
            xdimension="frequency",
            ydimension="toroidal_angle",
            method="pchip",
            coords={"poloidal_angle": self.poloidal_angles[0]},
        )

        opt_tor_freq_spline = self.get_opt_tor_freq_spline(
            descriptor, opt_tor_type=opt_tor_type
        )

        R_spline = lambda freq: cutoff_R_spline(
            (
                freq,
                opt_tor_freq_spline(freq),
            )
        )

        Z_spline = lambda freq: cutoff_Z_spline(
            (
                freq,
                opt_tor_freq_spline(freq),
            )
        )

        self.ds_dict[descriptor]["cutoff_R"] = xr.DataArray(
            data=R_spline(self.frequencies),
            dims=("frequency"),
            coords={"frequency": self.frequencies},
        )
        self.ds_dict[descriptor]["cutoff_Z"] = xr.DataArray(
            data=Z_spline(self.frequencies),
            dims=("frequency"),
            coords={"frequency": self.frequencies},
        )

        return R_spline(self.frequencies), R_spline(self.frequencies)

    def get_interp_variables(self, SweepDataset):
        """Reads the variables opt_tor, delta_theta_m, K_magnitude from the SweepDataset.
        Only data corresponding to the poloidal steering, toroidal range and frequencies
        of the PitchDiagnostic is read and stored.

        Args:
            SweepDataset (SweepDataset): SweepDataset to read from.
        """
        descriptor = str(SweepDataset.descriptor)
        if descriptor in self.ds_dict:
            ds = self.ds_dict[descriptor]
            if {
                "opt_tor_mismatch",
                "opt_tor_loc_m",
                "opt_tor_loc_product",
                "delta_theta_m",
                "K_magnitude",
            }.issubset(set(ds.keys())):
                print(f"Variables already interpolated from {descriptor}!")
                return None
        else:
            self.ds_dict[descriptor] = xr.Dataset()

        for key in ["mismatch", "loc_m", "loc_product"]:
            opt_tor_spline = SweepDataset.create_2Dspline(
                variable=f"opt_tor_{key}",
                xdimension="frequency",
                ydimension="poloidal_angle",
                method="pchip",
            )
            print(key)
            x, y = self.frequencies, self.poloidal_angles
            x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
            opt_tor_array = opt_tor_spline((x_grid, y_grid))
            opt_tor_da = xr.DataArray(
                data=opt_tor_array,
                dims=("frequency", "poloidal_angle"),
                coords={
                    "frequency": x,
                    "poloidal_angle": y,
                },
            )
            self.ds_dict[descriptor][f"opt_tor_{key}"] = opt_tor_da
            freq_range = np.linspace(self.frequencies[0], self.frequencies[-1], 500)
            X1, Y1 = np.meshgrid(freq_range, self.poloidal_angles, indexing="ij")
            opt_tor_profile = opt_tor_spline((X1, Y1))
            self.ds_dict[descriptor][f"opt_tor_{key}_profile"] = xr.DataArray(
                data=opt_tor_profile,
                dims=("freq_range", "poloidal_angle"),
                coords={
                    "freq_range": freq_range,
                    "poloidal_angle": self.poloidal_angles,
                },
            )

        delta_m_spline = SweepDataset.create_3Dspline(
            variable="cutoff_delta_theta_m",
            xdimension="frequency",
            ydimension="poloidal_angle",
            zdimension="toroidal_angle",
        )
        X, Y, Z = np.meshgrid(
            self.frequencies, self.poloidal_angles, self.toroidal_angles, indexing="ij"
        )
        delta_m_array = delta_m_spline((X, Y, Z))
        delta_theta_m_da = xr.DataArray(
            data=delta_m_array,
            dims=("frequency", "poloidal_angle", "toroidal_angle"),
            coords={
                "frequency": self.frequencies,
                "poloidal_angle": self.poloidal_angles,
                "toroidal_angle": self.toroidal_angles,
            },
        )
        self.ds_dict[descriptor]["delta_theta_m"] = delta_theta_m_da

        K_magnitude_spline = SweepDataset.create_3Dspline(
            variable="cutoff_K_magnitude_array",
            xdimension="frequency",
            ydimension="poloidal_angle",
            zdimension="toroidal_angle",
        )

        K_magnitude_array = K_magnitude_spline((X, Y, Z))
        K_magnitude_da = xr.DataArray(
            data=K_magnitude_array,
            dims=("frequency", "poloidal_angle", "toroidal_angle"),
            coords={
                "frequency": self.frequencies,
                "poloidal_angle": self.poloidal_angles,
                "toroidal_angle": self.toroidal_angles,
            },
        )
        self.ds_dict[descriptor]["K_magnitude"] = K_magnitude_da

        return None

    def get_cutoff_pitch(self, SweepDataset):
        """Reads the cutoff_pitch variable from the SweepDataset. Only data
        corresponding to the poloidal steering, toroidal range and frequencies
        of the PitchDiagnostic is read and stored. The SweepDataset must have
        the topfiles loaded and the method generate_pitch_at_cutoff called.

        Args:
            SweepDataset (SweepDataset): SweepDataset to read from.
        """
        descriptor = str(SweepDataset.descriptor)
        if descriptor in self.ds_dict:
            ds = self.ds_dict[descriptor]
            if {"cutoff_pitch"}.issubset(set(ds.keys())):
                print(f"Variables already interpolated from {descriptor}!")
                return None
        else:
            self.ds_dict[descriptor] = xr.Dataset()
        # Get the topfile path if available
        try:
            topfile = SweepDataset.get_topfile_path()
            self.set_topfile_path(topfile, descriptor)
            self.from_topfile_path(topfile, descriptor)
        except ValueError as error:
            print(
                f"{descriptor} does not have topfile string assigned. Error message: ",
                error,
            )

        try:
            pitch_spline = SweepDataset.create_3Dspline(
                variable="cutoff_pitch",
                xdimension="frequency",
                ydimension="poloidal_angle",
                zdimension="toroidal_angle",
            )
            X, Y, Z = np.meshgrid(
                self.frequencies,
                self.poloidal_angles,
                self.toroidal_angles,
                indexing="ij",
            )
            pitch_array = pitch_spline((X, Y, Z))
            pitch_da = xr.DataArray(
                data=pitch_array,
                dims=("frequency", "poloidal_angle", "toroidal_angle"),
                coords={
                    "frequency": self.frequencies,
                    "poloidal_angle": self.poloidal_angles,
                    "toroidal_angle": self.toroidal_angles,
                },
            )
            self.ds_dict[descriptor]["cutoff_pitch"] = pitch_da
            return pitch_da
        except ValueError as error:
            print(
                f"{descriptor} does not have cutoff_pitch data. Error message: ", error
            )

    def simulate_loc_measurements(self, SweepDataset, iterations=500, std_noise=None):
        """Simulate noisy measurements taken by the diagnostic array. The backscattered
        power is calculated separately for both integrated loc_m and loc_product.

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

        if SweepDataset.descriptor in self.ds_dict.keys():
            ds = self.ds_dict[descriptor]
            if {"simulated_loc_m", "simulated_loc_product"}.issubset(ds.keys()):
                print("Simulation data already present!")
                return ds["simulated_loc_m"], ds["simulated_loc_product"]
        else:
            self.ds_dict[descriptor] = xr.Dataset()
            ds = self.ds_dict[descriptor]

        self.ds_dict[descriptor]["loc_m_profile"] = xr.DataArray(
            dims=("frequency", "poloidal_angle", "toroidal_range"),
            coords={
                "frequency": frequencies,
                "poloidal_angle": poloidal_angles,
                "toroidal_range": toroidal_range,
            },
        )

        self.ds_dict[descriptor]["loc_product_profile"] = xr.DataArray(
            dims=("frequency", "poloidal_angle", "toroidal_range"),
            coords={
                "frequency": frequencies,
                "poloidal_angle": poloidal_angles,
                "toroidal_range": toroidal_range,
            },
        )

        self.ds_dict[descriptor]["simulated_loc_m"] = xr.DataArray(
            dims=("frequency", "poloidal_angle", "toroidal_angle", "iteration"),
            coords={
                "frequency": frequencies,
                "poloidal_angle": poloidal_angles,
                "toroidal_angle": toroidal_angles,
                "iteration": iter_array,
            },
        )

        self.ds_dict[descriptor]["simulated_loc_product"] = xr.DataArray(
            dims=("frequency", "poloidal_angle", "toroidal_angle", "iteration"),
            coords={
                "frequency": frequencies,
                "poloidal_angle": poloidal_angles,
                "toroidal_angle": toroidal_angles,
                "iteration": iter_array,
            },
        )

        for combination in product(frequencies, poloidal_angles):
            frequency, poloidal_angle = combination
            coords = {
                "frequency": frequency,
                "poloidal_angle": poloidal_angle,
            }
            int_loc_m_spline = SweepDataset.create_1Dspline(
                variable="int_loc_m",
                dimension="toroidal_angle",
                coords=coords,
            )
            int_loc_product_spline = SweepDataset.create_1Dspline(
                variable="int_loc_product",
                dimension="toroidal_angle",
                coords=coords,
            )
            loc_m_signal = int_loc_m_spline(toroidal_range)
            loc_product_signal = int_loc_product_spline(toroidal_range)

            self.ds_dict[descriptor]["loc_m_profile"].loc[coords] = loc_m_signal
            self.ds_dict[descriptor]["loc_product_profile"].loc[
                coords
            ] = loc_product_signal

            for iteration in iter_array:
                loc_m_array = int_loc_m_spline(toroidal_angles)
                loc_product_array = int_loc_product_spline(toroidal_angles)
                loc_m_measurement = scale_array(loc_m_array) + np.random.normal(
                    0, std_noise, loc_m_array.shape
                )
                loc_product_measurement = scale_array(
                    loc_product_array
                ) + np.random.normal(0, std_noise, loc_product_array.shape)
                index = {
                    "frequency": frequency,
                    "poloidal_angle": poloidal_angle,
                    "iteration": iteration,
                }
                self.ds_dict[descriptor]["simulated_loc_m"].loc[
                    index
                ] = loc_m_measurement
                self.ds_dict[descriptor]["simulated_loc_product"].loc[
                    index
                ] = loc_product_measurement

        return (
            self.ds_dict[descriptor]["simulated_loc_m"],
            self.ds_dict[descriptor]["simulated_loc_product"],
        )

    def simulate_mismatch_measurements(
        self, SweepDataset, iterations=500, std_noise=None, noise_flag=True
    ):
        """Simulates backscattered power with noise received by antenna array based on the mismatch
        angle and mismatch tolerance, and stores it in a dataset, with the index variable set by the
        SweepDataset descriptor. Data is saved with the key 'simulated_mismatch'.

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
            if "simulated_mismatch" in sim_ds.keys():
                print("Simulation data already present!")
                return sim_ds["simulated_mismatch"]

        else:
            self.ds_dict[descriptor] = xr.Dataset()
            sim_ds = self.ds_dict[descriptor]

        sim_ds["simulated_mismatch"] = xr.DataArray(
            dims=("frequency", "poloidal_angle", "toroidal_angle", "iteration"),
            coords={
                "frequency": frequencies,
                "poloidal_angle": poloidal_angles,
                "toroidal_angle": toroidal_angles,
                "iteration": iter_array,
            },
        )
        sim_ds["mismatch_profile"] = xr.DataArray(
            dims=("frequency", "poloidal_angle", "toroidal_range"),
            coords={
                "frequency": frequencies,
                "poloidal_angle": poloidal_angles,
                "toroidal_range": toroidal_range,
            },
        )

        da = self.ds_dict[descriptor]["simulated_mismatch"]
        db = self.ds_dict[descriptor]["mismatch_profile"]

        for frequency in frequencies:
            for poloidal_angle in poloidal_angles:
                simulated_power_prof = SweepDataset.get_simulated_power_prof(
                    frequency,
                    poloidal_angle,
                    std_noise=std_noise,
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

        self.ds_dict[descriptor]["simulated_mismatch"] = da

        return da

    def simulate_all(self, SweepDataset):
        self.get_interp_variables(SweepDataset)
        self.simulate_mismatch_measurements(SweepDataset)
        self.simulate_loc_measurements(SweepDataset)
        self.get_cutoff_locations(SweepDataset)
        self.get_cutoff_pitch(SweepDataset)

    ## Analysis methods

    def fit_measurement_gaussians(
        self, descriptor=None, opt_tor_guess=-1, width_guess=4
    ):
        if descriptor:
            descriptors = [descriptor]
        else:
            descriptors = self.descriptors
        for desc in descriptors:
            if "mismatch_gaussian_coeffs" in self.ds_dict[desc]:
                print(f"Gaussian fits already generated for {desc}")
                continue

            mismatch = self.ds_dict[desc]["simulated_mismatch"]
            mismatch_results = self._fit_gaussians(
                mismatch,
                desc,
                opt_tor_type="mismatch",
                opt_tor_guess=opt_tor_guess,
                width_guess=width_guess,
            )
            self.ds_dict[desc]["mismatch_gaussian_coeffs"] = mismatch_results[
                "curvefit_coefficients"
            ]
            self.ds_dict[desc]["mismatch_gaussian_cov"] = mismatch_results[
                "curvefit_covariance"
            ]

            loc_m = self.ds_dict[desc]["simulated_loc_m"]
            loc_m_results = self._fit_gaussians(
                loc_m,
                desc,
                opt_tor_type="loc_m",
                opt_tor_guess=opt_tor_guess,
                width_guess=width_guess,
            )
            self.ds_dict[desc]["loc_m_gaussian_coeffs"] = loc_m_results[
                "curvefit_coefficients"
            ]
            self.ds_dict[desc]["loc_m_gaussian_cov"] = loc_m_results[
                "curvefit_covariance"
            ]

            loc_product = self.ds_dict[desc]["simulated_loc_product"]
            loc_product_results = self._fit_gaussians(
                loc_product,
                desc,
                opt_tor_type="loc_product",
                opt_tor_guess=opt_tor_guess,
                width_guess=width_guess,
            )
            self.ds_dict[desc]["loc_product_gaussian_coeffs"] = loc_product_results[
                "curvefit_coefficients"
            ]
            self.ds_dict[desc]["loc_product_gaussian_cov"] = loc_product_results[
                "curvefit_covariance"
            ]

    def _fit_gaussians(
        self, data, descriptor, opt_tor_type="mismatch", opt_tor_guess=-1, width_guess=4
    ):
        dic = {"opt_tor": opt_tor_guess, "width": width_guess}
        p0 = {key: value for key, value in dic.items() if value is not None}

        curvefit_results = data.curvefit(
            coords="toroidal_angle",
            func=fit_gaussian,
            p0=p0,
            skipna=True,
            errors="ignore",
        )
        curvefit_results["curvefit_coefficients"].attrs[
            "method"
        ] = "PitchDiagnostic.fit_gaussians"
        curvefit_results["curvefit_coefficients"].attrs["descriptor"] = descriptor
        curvefit_results["curvefit_coefficients"].attrs["fit_parameters"] = (
            f"opt_tor_{opt_tor_type}",
            "width",
            "amplitude",
        )
        return curvefit_results

    def aggregate_fitted_gaussians(self, descriptor=None):
        if descriptor:
            descriptors = [descriptor]
        else:
            descriptors = self.descriptors

        for descriptor in descriptors:
            if "mismatch_mean_opt_tor" in self.ds_dict[descriptor]:
                print(f"Gaussian fits already aggregated for {descriptor}")
                continue

            for string in ("mismatch", "loc_m", "loc_product"):
                coefficients = self.ds_dict[descriptor][f"{string}_gaussian_coeffs"]
                mean_opt_tor = coefficients.mean(dim="iteration", skipna=True)
                std_opt_tor = coefficients.std(dim="iteration", skipna=True, ddof=1)
                self.ds_dict[descriptor][f"{string}_mean_opt_tor"] = mean_opt_tor.isel(
                    param=0
                )
                self.ds_dict[descriptor][f"{string}_std_opt_tor"] = std_opt_tor.isel(
                    param=0
                )
                self.ds_dict[descriptor][f"{string}_mean_width"] = mean_opt_tor.isel(
                    param=1
                )
                self.ds_dict[descriptor][f"{string}_std_width"] = std_opt_tor.isel(
                    param=1
                )

    def get_pitch_relation_across_equilibria(self, opt_tor_type="mismatch"):
        for descriptor in self.descriptors:
            if "current_scaling" not in self.ds_dict[descriptor].attrs.keys():
                raise KeyError(
                    f"{descriptor} is not yet assigned a numerical current_scaling attribute."
                )

        das = [self.ds_dict[descriptor] for descriptor in self.descriptors]
        das.sort(key=lambda da: da.attrs["current_scaling"])
        opt_tor_das = [da[f"opt_tor_{opt_tor_type}"] for da in das]
        current_scalings = np.array([da.attrs["current_scaling"] for da in das])

        rho_array = np.linspace(0, 1, 100)
        ds = xr.Dataset()

        ds["opt_tor_freq"] = xr.concat(objs=opt_tor_das, dim=("current_scaling"))
        ds["opt_tor_freq"] = ds["opt_tor_freq"].assign_coords(
            coords={"current_scaling": current_scalings}
        )
        ds["opt_tor_rho"] = xr.DataArray(
            dims=("current_scaling", "rho"),
            coords={"current_scaling": current_scalings, "rho": rho_array},
        )
        ds["pitch_rho"] = xr.DataArray(
            dims=("current_scaling", "rho"),
            coords={"current_scaling": current_scalings, "rho": rho_array},
        )
        counter = 0
        for descriptor in self.descriptors:
            current_scaling = current_scalings[counter]
            opt_tor_spline = self.get_opt_tor_rho_spline(descriptor)
            pitch_rho_spline = self.get_pitch_rho_spline(descriptor)
            opt_tor_array = opt_tor_spline(rho_array)
            pitch_rho_array = pitch_rho_spline(rho_array)
            ds["opt_tor_rho"].loc[{"current_scaling": current_scaling}] = opt_tor_array
            ds["pitch_rho"].loc[{"current_scaling": current_scaling}] = pitch_rho_array
            counter += 1
        self.pitch_ds = ds
        return ds

    def linregress_pitch_rho(self):
        opt_tor_da = self.pitch_ds["opt_tor_rho"]
        pitch_da = self.pitch_ds["pitch_rho"]
        rho_coords = self.pitch_ds.coords["rho"].values

        ds = self.pitch_ds
        for key in ("gradient", "intercept", "rval", "stderr"):
            ds[key] = xr.DataArray(dims=("rho",), coords={"rho": rho_coords})

        for rho in rho_coords:
            opt_tor = opt_tor_da.loc[{"rho": rho}]
            pitch = pitch_da.loc[{"rho": rho}]
            result = linregress(opt_tor, pitch)
            ds["gradient"].loc[{"rho": rho}] = result.slope
            ds["intercept"].loc[{"rho": rho}] = result.intercept
            ds["rval"].loc[{"rho": rho}] = result.rvalue
            ds["stderr"].loc[{"rho": rho}] = result.stderr

        return ds["gradient"], ds["intercept"]

    def analyse_all(self, opt_tor_guess=-1, width_guess=4):
        self.fit_measurement_gaussians(
            opt_tor_guess=opt_tor_guess, width_guess=width_guess
        )
        self.aggregate_fitted_gaussians()
        try:
            for key in ["mismatch", "loc_m", "loc_product"]:
                self.get_pitch_relation_across_equilibria(opt_tor_type=key)
            self.linregress_pitch_rho()
        except Exception:
            print(
                "Methods get_pitch_relation_across_equilibria() and linregress_pitch_rho() failed. Check if scaling attributes are assigned."
            )

    ## Plot methods

    def plot_profile_comparisons(self, descriptor):
        """Plots the backscattered power profiles as calculated by the mismatch gaussian
        formula, and by integrating loc_m and loc_product.

        Args:
            decriptor(str): Descriptor of the Dataset to plot.

        Returns:
            fig, axes
        """
        mismatch_profile = self.ds_dict[descriptor]["mismatch_profile"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        loc_m_profile = self.ds_dict[descriptor]["loc_m_profile"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        loc_product_profile = self.ds_dict[descriptor]["loc_product_profile"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]

        toroidal_range = mismatch_profile.coords["toroidal_range"].values
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
        fig, axes = plt.subplots(
            rows, cols, dpi=150, figsize=(5, 7), sharex=True, sharey=True
        )
        fig.tight_layout()
        for frequency in self.frequencies:
            row, col = combinations[counter]
            ax = axes[row, col]

            ax.plot(
                toroidal_range,
                mismatch_profile[counter],
                color="k",
                label="mismatch formula",
            )
            ax.plot(
                toroidal_range,
                scale_array(loc_m_profile[counter]),
                color="r",
                label="integrated loc_m",
            )
            ax.plot(
                toroidal_range,
                scale_array(loc_product_profile[counter]),
                color="b",
                label="integrated loc_product",
            )
            ax.set_title(f"{frequency} GHz", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=5)
            counter += 1

        ax.legend(
            bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0.0, fontsize=8
        )
        fig.suptitle("Simulated profile comparison", fontsize=15)
        plt.subplots_adjust(top=0.90)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        plt.xlabel("Toroidal Steering/$^\circ$")
        plt.ylabel("Normalized Received Power")
        return fig, axes

    def plot_profile_fits(self, descriptor, sim_type="mismatch"):
        """Plots the mean gaussian fit over the original backscattered power profile
        for the set of frequencies defined when intializing the class.

        Args:
            descriptor (str): Descriptor of the Dataset to plot.
            sim_type (str, optional): One of ('mismatch', 'loc_m', 'loc_product'). Defaults to 'mismatch'.

        Returns:
            fig, axes
        """
        data = self.ds_dict[descriptor][f"{sim_type}_profile"]
        opt_tor = self.ds_dict[descriptor][f"{sim_type}_mean_opt_tor"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        delta = self.ds_dict[descriptor][f"{sim_type}_mean_width"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        actual = self.ds_dict[descriptor][f"opt_tor_{sim_type}"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]

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
        fig, axes = plt.subplots(
            rows, cols, dpi=150, figsize=(5, 7), sharex=True, sharey=True
        )
        fig.tight_layout()

        for frequency in self.frequencies:
            row, col = combinations[counter]
            ax = axes[row, col]
            color = "k"

            lin_data = data.loc[
                {"frequency": frequency, "poloidal_angle": self.poloidal_angles[0]}
            ]
            opt_tor_val = opt_tor.loc[{"frequency": frequency}].values
            delta_val = delta.loc[{"frequency": frequency}].values
            gaussian_profile = fit_gaussian(toroidal_range, opt_tor_val, delta_val)

            ax.plot(
                toroidal_range,
                scale_array(lin_data),
                color=color,
                label="profile",
            )
            ax.axvline(
                actual.loc[{"frequency": frequency}],
                label="predicted opt_tor",
                color=color,
            )
            ax.plot(
                toroidal_range,
                gaussian_profile,
                color="r",
                linestyle="--",
                label="mean gaussian fit",
            )
            ax.axvline(
                opt_tor_val,
                label="mean gaussian-fitted peak",
                color="r",
                linestyle="--",
            )
            ax.set_title(f"{frequency} GHz", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=5)
            counter += 1

        ax.legend(
            bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0.0, fontsize=8
        )
        fig.suptitle(f"{sim_type} original profile vs. gaussian fit", fontsize=15)
        plt.subplots_adjust(top=0.90)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        plt.xlabel("Toroidal Steering/$^\circ$")
        plt.ylabel("Normalized Received Power")
        return fig, axes

    def plot_opt_tor_hist(self, descriptor, sim_type="mismatch"):
        """Plots a series of histograms of the opt tor fits generated by the fit_measurement_gaussians
        method.

        Args:
            descriptor (str): Descriptor of the Dataset to plot.
            sim_type (str, optional): One of ('mismatch', 'loc_m', 'loc_product'). Defaults to 'mismatch'.

        Returns:
            fig, axes
        """
        data = self.ds_dict[descriptor][f"{sim_type}_profile"]
        toroidal_range = data.coords["toroidal_range"].values

        mean = self.ds_dict[descriptor][f"{sim_type}_mean_opt_tor"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        actual = self.ds_dict[descriptor][f"opt_tor_{sim_type}"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]

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

        data = self.ds_dict[descriptor][f"{sim_type}_gaussian_coeffs"].isel(param=0)
        counter = 0
        combinations = list(product(range(rows), range(cols)))

        fig, axes = plt.subplots(
            rows, cols, dpi=150, figsize=(5, 7), sharex=True, sharey=True
        )
        fig.tight_layout()
        for frequency in self.frequencies:
            row, col = combinations[counter]
            ax = axes[row, col]
            data_slice = data.loc[
                {"frequency": frequency, "poloidal_angle": self.poloidal_angles[0]}
            ]
            color = colormap(counter / freq_count)
            ax.hist(data_slice, alpha=0.5, color=color)
            ax.axvline(
                actual.loc[{"frequency": frequency}], label="predicted", color="k"
            )
            ax.axvline(
                mean.loc[{"frequency": frequency}],
                label="mean",
                color="r",
                linestyle="--",
            )
            ax.set_title(f"{frequency} GHz", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=5)
            ax.set_xlim(toroidal_range[0], toroidal_range[-1])

            # ax.legend()
            counter += 1
        ax.legend(
            bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0.0, fontsize=8
        )
        fig.suptitle(f"{sim_type} opt_tor fits with noise", fontsize=15)
        plt.subplots_adjust(top=0.90)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        plt.xlabel("Measured optimum toroidal steering/$^\circ$")
        plt.ylabel("Number of measurements")

        return fig, axes

    def plot_variable_vs_tor(self, variable, descriptor, opt_tor_type="mismatch"):
        """Plot how a variable changes over the toroidal steering range, normalized
        to the value at opt_tor.

        Args:
            variable (str): One of ("pitch_angle", "K_magnitude", "delta_theta_m")
            descriptor (str): Descriptor of the Dataset to plot.

        Returns:
            fig, axes
        """
        data = self.ds_dict[descriptor][variable]
        actual = self.ds_dict[descriptor][f"opt_tor_{opt_tor_type}"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]

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
        fig, axes = plt.subplots(
            rows, cols, dpi=150, figsize=(5, 7), sharex=True, sharey=True
        )
        fig.tight_layout()

        for frequency in self.frequencies:
            row, col = combinations[counter]
            ax = axes[row, col]
            opt_tor = actual.loc[{"frequency": frequency}].values
            opt_tor_val = data.sel(
                {
                    "frequency": frequency,
                    "poloidal_angle": self.poloidal_angles[0],
                    "toroidal_angle": opt_tor,
                },
                method="nearest",
            ).values
            ax.plot(
                self.toroidal_angles,
                data.loc[
                    {"frequency": frequency, "poloidal_angle": self.poloidal_angles[0]}
                ]
                / opt_tor_val,
                label=f"{variable}",
            )
            ax.axvline(opt_tor, label=f"opt_tor_{opt_tor_type}", color="k")
            ax.set_title(f"{frequency} GHz", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=5)

            counter += 1

        ax.legend(
            bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0.0, fontsize=8
        )
        fig.suptitle(f"Normalized {variable} vs. Toroidal Steering", fontsize=15)
        plt.subplots_adjust(top=0.90)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        plt.xlabel("Toroidal steering/$^\circ$")
        plt.ylabel(f"{variable}")
        return fig, axes

    def plot_all_opt_tor_fits(self, sim_type="mismatch"):
        """Plots the optimal toroidal steering vs. frequency for all loaded SweepDatasets, with
        colored region representing 1 standard deviation of fitted optimal toroidal steerings.

        Args:
            sim_type (str, optional): One of ('mismatch', 'loc_m', 'loc_product'). Defaults to 'mismatch'.

        Returns:
            fig
        """
        fig = plt.figure()
        plt.title(f"{sim_type}, Opt Tor vs. Probe Frequency", fontsize=15)
        colormap = plt.cm.gnuplot2
        counter = 0
        total_count = len(self.descriptors)

        for descriptor in self.descriptors:
            color = colormap(counter / total_count)
            mean = self.ds_dict[descriptor][f"{sim_type}_mean_opt_tor"].loc[
                {"poloidal_angle": self.poloidal_angles[0]}
            ]
            std = self.ds_dict[descriptor][f"{sim_type}_std_opt_tor"].loc[
                {"poloidal_angle": self.poloidal_angles[0]}
            ]
            actual = self.ds_dict[descriptor][f"opt_tor_{sim_type}"].loc[
                {"poloidal_angle": self.poloidal_angles[0]}
            ]
            x_coords = self.frequencies
            plt.plot(x_coords, mean, color=color, marker="+")
            plt.fill_between(x_coords, mean - std, mean + std, color=color, alpha=0.3)
            plt.plot(x_coords, actual, color=color, marker="+", linestyle="dashed")
            counter += 1

        Line2D = mpl.lines.Line2D
        custom_colors = [
            Line2D([0], [0], color=colormap(count / total_count), lw=4)
            for count in range(total_count)
        ]
        custom_styles = [
            Line2D([0], [0], color="k", linestyle="solid"),
            Line2D([0], [0], color="k", linestyle="dashed"),
        ]
        custom_handles = custom_colors + custom_styles
        custom_labels = list(self.descriptors) + ["mean", "predicted"]

        plt.xlabel("Frequency/GHz")
        plt.ylabel("Opt. Toroidal Steering/$^\circ$")
        plt.legend(
            custom_handles,
            custom_labels,
            bbox_to_anchor=(1.05, 0),
            loc="lower left",
            borderaxespad=0.0,
        )
        return fig

    def plot_opt_tor_with_widths(self, sim_type="mismatch"):
        """Plots the optimal toroidal steering vs. frequency for all loaded SweepDatasets, with
        dotted lines marking out the full-width half-maximum of the gaussian-fitted backscatter
        profile.

        Args:
            sim_type (str, optional): One of ('mismatch', 'loc_m', 'loc_product'). Defaults to 'mismatch'.

        Returns:
            fig
        """
        fig = plt.figure()
        plt.title(
            f"{sim_type}, Optimal toroidal steering and profile widths", fontsize=15
        )
        colormap = plt.cm.gnuplot2
        counter = 0
        total_count = len(self.descriptors)
        data = self.ds_dict[descriptor][f"{sim_type}_profile"]
        actual = self.ds_dict[descriptor][f"opt_tor_{sim_type}"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]

        for descriptor in self.descriptors:
            color = colormap(counter / total_count)
            mean = self.ds_dict[descriptor][f"{sim_type}_mean_opt_tor"].loc[
                {"poloidal_angle": self.poloidal_angles[0]}
            ]
            width = self.ds_dict[descriptor][f"{sim_type}_mean_width"].loc[
                {"poloidal_angle": self.poloidal_angles[0]}
            ]
            actual = self.ds_dict[descriptor][f"opt_tor_{sim_type}"].loc[
                {"poloidal_angle": self.poloidal_angles[0]}
            ]
            x_coords = self.frequencies
            plt.plot(x_coords, mean, color=color, marker="+")
            plt.plot(x_coords, mean + width / 2, color=color, linestyle=(0, (5, 10)))
            plt.plot(x_coords, mean - width / 2, color=color, linestyle=(0, (5, 10)))
            plt.plot(x_coords, actual, color=color, marker="+", linestyle="dashed")
            counter += 1

        for toroidal_angle in self.toroidal_angles:
            plt.axhline(toroidal_angle, linestyle="dotted", alpha=0.5)
        Line2D = mpl.lines.Line2D
        custom_colors = [
            Line2D([0], [0], color=colormap(count / total_count), lw=4)
            for count in range(total_count)
        ]
        custom_styles = [
            Line2D([0], [0], color="k", linestyle="solid"),
            Line2D([0], [0], color="k", linestyle="dashed"),
            Line2D([0], [0], color="k", linestyle=(0, (5, 10))),
            Line2D([0], [0], color="k", linestyle="dotted"),
        ]
        custom_handles = custom_colors + custom_styles
        custom_labels = list(self.descriptors) + [
            "mean",
            "predicted",
            "FWHM bounds",
            "antenna steerings",
        ]

        plt.xlabel("Frequency/GHz")
        plt.ylabel("Opt. Toroidal Steering/$^\circ$")
        plt.legend(
            custom_handles,
            custom_labels,
            bbox_to_anchor=(1.05, 0),
            loc="lower left",
            borderaxespad=0.0,
        )
        return fig

    def plot_opt_tor_vs_freq(self, descriptor, sim_type="mismatch"):
        """Plots the optimal toroidal steering vs. frequency for a specified Dataset, with
        colored region representing 1 standard deviation of fitted optimal toroidal steerings.

        Args:
            descriptor: Descriptor of dataset to plot for
            sim_type (str, optional): One of ('mismatch', 'loc_m', 'loc_product'). Defaults to 'mismatch'.

        Returns:
            fig
        """
        mean = self.ds_dict[descriptor][f"{sim_type}_mean_opt_tor"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        std = self.ds_dict[descriptor][f"{sim_type}_std_opt_tor"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        actual = self.ds_dict[descriptor][f"opt_tor_{sim_type}"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        x_coords = self.frequencies
        fig = plt.figure()
        plt.title(f"{sim_type}, Opt Tor vs. Probe Frequency")
        plt.plot(x_coords, mean, label="mean", marker="+")
        plt.fill_between(x_coords, mean - std, mean + std, label="1$\sigma$", alpha=0.3)
        plt.plot(x_coords, actual, label="predicted", color="r", marker="+")
        plt.xlabel("Frequency/GHz")
        plt.ylabel("Opt. Toroidal Steering/$^\circ$")
        plt.legend()
        return fig

    def plot_rho_freq_relation(self):
        rho_freq = self.get_rho_freq_spline()
        rho_upper, rho_lower = self.get_rho_delta_splines()
        freq_range = np.linspace(self.frequencies[0], self.frequencies[-1], 500)

        colormap = plt.cm.gnuplot2
        counter = 0

        # X, Y = np.meshgrid()

        fig = plt.figure()
        plt.plot(
            freq_range, rho_freq(freq_range), label="mean rho", color="r", linewidth=0.2
        )
        plt.fill_between(
            freq_range,
            rho_upper(freq_range),
            rho_lower(freq_range),
            label="delta rho",
            color="b",
            alpha=0.2,
        )

        for frequency in self.frequencies:
            color = colormap(counter / len(self.frequencies))
            plt.axvline(frequency, color=color, linewidth=0.2, linestyle="--")
            upper_array = rho_upper(frequency) * np.ones_like(freq_range)
            lower_array = rho_lower(frequency) * np.ones_like(freq_range)
            delta = np.format_float_scientific(
                rho_upper(frequency) - rho_lower(frequency), precision=1
            )
            plt.fill_between(
                freq_range,
                upper_array,
                lower_array,
                color=color,
                alpha=0.2,
                label=f"{frequency}GHz $\Delta$rho = {delta}",
            )
            counter += 1

        plt.xlabel("Frequency/GHz")
        plt.ylabel("Cutoff Rho")
        plt.title("Mean $\\rho(f)$ and $\Delta\\rho$ over toroidal steering range")
        plt.legend(bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0.0)
        return fig

    def plot_cutoff_positions(self, xlim=None, ylim=None, levels=20, cmap="plasma_r"):
        """Plots the cutoff positions reached by the antenna in the plasma in R-Z coordinates
        for all datasets, against a set of poloidal flux contours.
        """
        flag = True
        fig, ax = plt.subplots()
        markers = ["o", "X", "v", "^", "s", "D", "p", "*", "h", "8", "+"]
        mcount = 0

        for descriptor in self.descriptors:
            ds = self.ds_dict[descriptor]
            marker = markers[mcount]
            if flag:
                rho = np.sqrt(self.topfiles[descriptor]["pol_flux"].transpose("Z", "R"))
                cont = rho.plot.contour(
                    levels=levels,
                    vmax=1.0,
                    xlim=xlim,
                    ylim=ylim,
                    cmap=cmap,
                )
                ax.clabel(cont, inline=True, fontsize=5)
                flag = False
            try:
                cutoff_R = ds["cutoff_R"]
                cutoff_Z = ds["cutoff_Z"]
                sc = ax.scatter(
                    cutoff_R,
                    cutoff_Z,
                    s=10,
                    c=self.frequencies,
                    cmap="cool",
                    edgecolors="k",
                    linewidths=0.5,
                    marker=marker,
                )
                mcount += 1
            except KeyError:
                print(f"No cutoff position data for {descriptor}.")
                continue

        plt.colorbar(sc, label="Freq/GHz")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("R/m")
        ax.set_ylabel("Z/m")
        fig.suptitle("Cutoff positions vs. scaling", fontsize=15)

        Line2D = mpl.lines.Line2D
        custom_handles = [
            Line2D([0], [0], color="k", marker=markers[i], linestyle="None")
            for i in range(len(self.descriptors))
        ]
        custom_labels = list(self.descriptors)
        ax.legend(
            custom_handles,
            custom_labels,
            bbox_to_anchor=(1.25, 0),
            loc="lower left",
            borderaxespad=0.0,
        )
        return fig, ax

    def plot_pitch_contours(self, descriptor, xlim=None, ylim=None, vmin=-20):
        """Plots the pitch angle on the R-Z plane as a set of filled contours against
        a foreground of poloidal flux contour lines. Cutoff positions reached by
        the diagnostic antennas are also plotted.
        """
        topfile = self.topfiles[descriptor]
        ds = self.ds_dict[descriptor]

        fig, ax = plt.subplots()
        topfile["pitch_angle"].transpose().plot.contourf(
            levels=30,
            add_colorbar=True,
            xlim=xlim,
            ylim=ylim,
            cmap="plasma",
            vmin=vmin,
        )
        cont = np.sqrt(topfile["pol_flux"].transpose()).plot.contour(
            levels=15,
            vmax=1.0,
            xlim=xlim,
            ylim=ylim,
            colors="k",
        )
        ax.clabel(cont, inline=True, fontsize=5)

        cutoff_R = ds["cutoff_R"]
        cutoff_Z = ds["cutoff_Z"]
        sc = ax.scatter(
            cutoff_R,
            cutoff_Z,
            s=15,
            c=self.frequencies,
            cmap="Spectral",
            marker="o",
            label="Freq/GHz",
        )
        plt.gca().set_aspect(1.0)
        fig.colorbar(sc, label="Freq/GHz")
        fig.suptitle("Pitch angle contours and cutoff locations", fontsize=15)
        return fig, ax

    def plot_pitch_angle_vs_opt_tor(
        self, rho_lower=0.1, rho_upper=1.0, unit="deg", sample_rho=0.5
    ):
        """Plots pitch angle against opt_tor for all equilibriums, and automatically
        calculates linregress-fitted slope for the range of rho set.

        Args:
            rho_lower (float): Lower range of rho for plotting and linear fitting
            rho_upper (float): Upper range of rho for plotting and linear fitting

        Returns:
            Artists (fig, ax), dictionary of slopes with descriptor keys
        """
        if unit not in ("deg", "rad"):
            raise ValueError("Invalid unit specified")

        rho_range = np.linspace(rho_lower, rho_upper, 500)
        sample_index = np.absolute(rho_range - sample_rho).argmin()
        fig, ax = plt.subplots()
        seq_cmaps = [
            "Greys",
            "Purples",
            "Blues",
            "Greens",
            "Oranges",
            "Reds",
            "RdPu",
            "BuPu",
            "GnBu",
            "YlGn",
            "YlOrBr",
        ]
        solid_cols = [
            "dimgray",
            "indigo",
            "royalblue",
            "green",
            "darkorange",
            "red",
            "deeppink",
            "mediumslateblue",
            "turquoise",
            "greenyellow",
            "yellow",
        ]
        counter = 0
        sampled_rho = rho_range[sample_index]
        print("Rho sampled (dotted red line): ", sampled_rho)
        sampled_pitch = []
        sampled_opt_tor = []
        for descriptor in self.descriptors:
            cmap = seq_cmaps[counter]
            color = solid_cols[counter]
            opt_tor_spline = self.get_opt_tor_rho_spline(descriptor)
            pitch_rho_spline = self.get_pitch_rho_spline(descriptor)
            opt_tor = opt_tor_spline(rho_range)
            pitch = pitch_rho_spline(rho_range)
            if unit == "rad":
                opt_tor = np.deg2rad(opt_tor)
                pitch = np.deg2rad(pitch)
            sampled_pitch.append(pitch[sample_index])
            sampled_opt_tor.append(opt_tor[sample_index])
            scat = ax.scatter(
                opt_tor, pitch, s=3, c=rho_range, cmap=cmap, label=f"{descriptor}"
            )
            if counter == 0:
                fig.colorbar(scat, label="$\\rho$")
            counter += 1
        ax.plot(
            sampled_opt_tor,
            sampled_pitch,
            linestyle="--",
            color="r",
            label=f"rho={round(sampled_rho, 2)}",
            linewidth=1,
        )
        ax.set_xlabel("opt_tor/$^\circ$")
        ax.set_ylabel("pitch angle/$^\circ$")
        if unit == "rad":
            ax.set_xlabel("opt_tor/rad")
            ax.set_ylabel("pitch angle/rad")
        Line2D = mpl.lines.Line2D
        custom_handles = [
            Line2D([0], [0], color=solid_cols[i]) for i in range(len(self.descriptors))
        ]
        custom_labels = list(self.descriptors)
        ax.legend(
            custom_handles,
            custom_labels,
            bbox_to_anchor=(1.25, 0),
            loc="lower left",
            borderaxespad=0.0,
        )
        fig.suptitle(
            f"Pitch angle relation, {rho_lower} < $\\rho$ < {rho_upper}", fontsize=15
        )
        return fig, ax

    def plot_pitch_angle_vs_opt_tor2(
        self,
        rho_lower=0.1,
        rho_upper=1.0,
        unit="deg",
        levels=20,
    ):
        """Plots pitch angle against opt_tor at a sampled set of rho positions
        across all equilibriums.

        Returns:
            fig, ax
        """

        opt_tor_da = self.pitch_ds["opt_tor_rho"]
        pitch_da = self.pitch_ds["pitch_rho"]
        rho_array = np.linspace(rho_lower, rho_upper, levels)
        cmap = plt.cm.plasma_r
        # Interpolating to fit with the range of rho
        opt_tor_interp = opt_tor_da.interp(coords={"rho": rho_array})
        pitch_interp = pitch_da.interp(coords={"rho": rho_array})

        fig, ax = plt.subplots()

        for rho in rho_array:
            opt_tor = opt_tor_interp.loc[{"rho": rho}]
            pitch = pitch_interp.loc[{"rho": rho}]
            if unit == "rad":
                opt_tor = np.deg2rad(opt_tor)
                pitch = np.deg2rad(pitch)
            ax.plot(opt_tor, pitch, color=cmap(rho), label=f"{rho}", marker="x")

        ax.set_xlabel("optimal toroidal steering/$^\circ$")
        ax.set_ylabel("pitch angle/$^\circ$")
        if unit == "rad":
            ax.set_xlabel("opt_tor/rad")
            ax.set_ylabel("pitch angle/rad")
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim(vmin=0, vmax=1)
        fig.colorbar(sm, label="$\\rho$")
        fig.suptitle(
            f"Pitch angle relation, {rho_lower} < $\\rho$ < {rho_upper}", fontsize=15
        )
        return fig, ax

    def plot_linear_fits_vs_rho(self, key, rho_min=0, rho_max=1):
        """Plots the results of a linear regression of pitch angle vs. optimal toroidal
        steering over a set of rho.

        Args:
            key (str): One of ('gradient', 'intercept', 'rval', 'std_err')
            rho_min (float): Lower bound of rho to plot
            rho_max (float): Upper bound of rho to plot

        Returns:
            fig, ax
        """
        fig, ax = plt.subplots()
        da = self.pitch_ds[key].loc[dict(rho=slice(rho_min, rho_max))]
        da.plot(color="k")
        fig.suptitle(f"{key} of pitch angle v. opt_tor over rho")
        return fig, ax

    def plot_delta_opt_tor_vs_scaling(self):
        """Plots how the amplitude of change in optimal toroidal angle varies with scaling.
        Requires a current scaling attribbute to be set for all datasets.

        Returns:
            fig, ax
        """
        da = self.pitch_ds["opt_tor_freq"]
        scaling = da.coords["current_scaling"].values
        cmap = plt.cm.plasma
        fig, ax = plt.subplots()
        slopes = []
        for frequency in self.frequencies:
            color = cmap(
                (frequency - self.frequencies[0])
                / (self.frequencies[-1] - self.frequencies[0])
            )
            opt_tors = da.loc[{"frequency": frequency}].squeeze()
            delta_opt_tors = opt_tors - opt_tors.loc[{"current_scaling": 1.0}].values
            result = linregress(scaling, delta_opt_tors)
            slopes.append(result.slope)
            ax.plot(
                scaling,
                delta_opt_tors,
                color=color,
                label=f"{frequency} GHz, m={round(result.slope, 2)}",
            )
        ax.legend(
            bbox_to_anchor=(1.05, 0),
            loc="lower left",
            borderaxespad=0.0,
        )
        ax.set_xlabel("scaling")
        ax.set_ylabel("$\Delta$opt_tor/$^\circ$")
        fig.suptitle("Change in optimum toroidal steering vs. scaling", fontsize=15)
        return fig, ax, np.array(slopes)

    def plot_delta_pitch_vs_scaling(self, default_descriptor="1.0"):
        """For the given array of frequencies, we assume a fixed set of probe locations within
        the plasma based on the 1.0X current scaling at optimal toroidal steering for this scaling.
        From this static set of 1.0X probe locations in R-Z coordinates, plot how the actual pitch
        angle changes at the location with current scaling.

        Returns:
            fig, ax, array: Array consists of linearly-fitted gradients as displayed in the legend.
        """

        cutoff_R = self.ds_dict[default_descriptor]["cutoff_R"]
        cutoff_Z = self.ds_dict[default_descriptor]["cutoff_Z"]
        scalings = self.pitch_ds["opt_tor_freq"].coords["current_scaling"].values
        da = xr.DataArray(
            dims=("frequency", "current_scaling"),
            coords={
                "frequency": self.frequencies,
                "current_scaling": scalings,
            },
        )
        for descriptor in self.descriptors:
            pitch_da = self.topfiles[descriptor]["pitch_angle"]
            pitch = pitch_da.sel(
                R=cutoff_R,
                Z=cutoff_Z,
                method="nearest",
            )
            scaling = self.ds_dict[descriptor].attrs["current_scaling"]
            da.loc[{"current_scaling": scaling}] = pitch

        slopes = []
        cmap = plt.cm.plasma
        fig, ax = plt.subplots()
        for frequency in self.frequencies:
            color = cmap(
                (frequency - self.frequencies[0])
                / (self.frequencies[-1] - self.frequencies[0])
            )
            pitch = da.loc[{"frequency": frequency}].squeeze()
            delta_pitch = pitch - pitch.loc[{"current_scaling": 1.0}].values
            result = linregress(scalings, delta_pitch)
            slopes.append(result.slope)
            ax.plot(
                scalings,
                delta_pitch,
                color=color,
                label=f"{frequency} GHz, m={round(result.slope, 2)}",
            )
        ax.legend(
            bbox_to_anchor=(1.05, 0),
            loc="lower left",
            borderaxespad=0.0,
        )
        ax.set_xlabel("scaling")
        ax.set_ylabel("$\Delta$pitch/$^\circ$")
        fig.suptitle(
            "Change in pitch angle vs. scaling fixed at 1.0X cutoff points", fontsize=15
        )
        return fig, ax, np.array(slopes)
