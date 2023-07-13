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

    Class is constructed by calling class methods create_new or from_netcdf.


    """

    def __init__(self, ds):
        self.home = ds
        self.ds_dict = {}

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

    def get_keys(self):
        keys_dict = {}
        for key in self.ds_dict.keys():
            values = self.ds_dict[key].keys()
            keys_dict[key] = tuple(values)
        print(keys)
        return keys_dict

    def set_std_noise(self, new_value):
        self.home.attrs["std_noise"] = new_value

    def set_rho_freq_relation(self, SweepDataset):
        # Assumed equilibrium relation for analysing all equilibria
        # Use an array representation of a spline
        # 2D spline contour averaged over toroidal range to get average relation
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

    def get_rho_freq_spline(self):
        frequency_range = np.linspace(self.frequencies[0], self.frequencies[-1], 500)
        return UnivariateSpline(
            frequency_range, self.ds_dict["cutoff_data"]["rho_freq_relation"], s=0
        )

    def get_rho_delta_splines(self):
        frequency_range = np.linspace(self.frequencies[0], self.frequencies[-1], 500)
        upper_spline = UnivariateSpline(
            frequency_range, self.ds_dict["cutoff_data"]["rho_upper"], s=0
        )
        lower_spline = UnivariateSpline(
            frequency_range, self.ds_dict["cutoff_data"]["rho_lower"], s=0
        )
        return upper_spline, lower_spline

    ## Simulation methods

    def get_interp_variables(self, SweepDataset):
        descriptor = str(SweepDataset.descriptor)
        if descriptor in self.ds_dict:
            ds = self.ds_dict[descriptor]
            if {"opt_tor", "delta_theta_m", "K_magnitude"}.issubset(set(ds.keys())):
                print(f"Variables already interpolated from {descriptor}!")
                return None
        else:
            self.ds_dict[descriptor] = xr.Dataset()

        opt_tor_spline = SweepDataset.create_2Dspline(
            variable="opt_tor", xdimension="frequency", ydimension="poloidal_angle"
        )
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
        self.ds_dict[descriptor]["opt_tor"] = opt_tor_da

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

    def simulate_loc_measurements(self, SweepDataset, iterations=500, std_noise=None):
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
            coords_dict = {
                "frequency": frequency,
                "poloidal_angle": poloidal_angle,
            }
            int_loc_m_spline = SweepDataset.create_1Dspline(
                variable="int_loc_m",
                dimension="toroidal_angle",
                coords_dict=coords_dict,
            )
            int_loc_product_spline = SweepDataset.create_1Dspline(
                variable="int_loc_product",
                dimension="toroidal_angle",
                coords_dict=coords_dict,
            )
            loc_m_signal = int_loc_m_spline(toroidal_range)
            loc_product_signal = int_loc_product_spline(toroidal_range)

            self.ds_dict[descriptor]["loc_m_profile"].loc[coords_dict] = loc_m_signal
            self.ds_dict[descriptor]["loc_product_profile"].loc[
                coords_dict
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

    ## Analysis methods

    def fit_measurement_gaussians(self, descriptor=None):
        if descriptor:
            descriptors = [descriptor]
        else:
            descriptors = self.descriptors
        for desc in descriptors:
            if "mismatch_gaussian_coeffs" in self.ds_dict[desc]:
                print(f"Gaussian fits already generated for {desc}")
                continue

            mismatch = self.ds_dict[desc]["simulated_mismatch"]
            mismatch_results = self._fit_gaussians(mismatch, desc)
            self.ds_dict[desc]["mismatch_gaussian_coeffs"] = mismatch_results[
                "curvefit_coefficients"
            ]
            self.ds_dict[desc]["mismatch_gaussian_cov"] = mismatch_results[
                "curvefit_covariance"
            ]

            loc_m = self.ds_dict[desc]["simulated_loc_m"]
            loc_m_results = self._fit_gaussians(loc_m, desc)
            self.ds_dict[desc]["loc_m_gaussian_coeffs"] = loc_m_results[
                "curvefit_coefficients"
            ]
            self.ds_dict[desc]["loc_m_gaussian_cov"] = loc_m_results[
                "curvefit_covariance"
            ]

            loc_product = self.ds_dict[desc]["simulated_loc_product"]
            loc_product_results = self._fit_gaussians(loc_product, desc)
            self.ds_dict[desc]["loc_product_gaussian_coeffs"] = loc_product_results[
                "curvefit_coefficients"
            ]
            self.ds_dict[desc]["loc_product_gaussian_cov"] = loc_product_results[
                "curvefit_covariance"
            ]

    def _fit_gaussians(self, data, descriptor):
        curvefit_results = data.curvefit(
            coords="toroidal_angle",
            func=fit_gaussian,
            p0={"opt_tor": -2},
            skipna=True,
            errors="ignore",
        )
        curvefit_results["curvefit_coefficients"].attrs[
            "method"
        ] = "PitchDiagnostic.fit_gaussians"
        curvefit_results["curvefit_coefficients"].attrs["descriptor"] = descriptor
        curvefit_results["curvefit_coefficients"].attrs["fit_parameters"] = (
            "opt_tor",
            "delta",
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
                self.ds_dict[descriptor][
                    f"{string}_mean_delta_theta_m"
                ] = mean_opt_tor.isel(param=1)
                self.ds_dict[descriptor][
                    f"{string}_std_delta_theta_m"
                ] = std_opt_tor.isel(param=1)

    ## Plot methods

    def plot_profile_comparisons(self, descriptor):
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
        plt.xlabel("toroidal steering/deg")
        plt.ylabel("normalized received power")
        return fig, axes

    def plot_profile_fits(self, descriptor, sim_type="mismatch"):
        """_summary_

        Args:
            descriptor (_type_): _description_
            sim_type (str, optional): One of ('mismatch', 'loc_m', 'loc_product'). Defaults to 'mismatch'.

        Returns:
            _type_: _description_
        """
        data = self.ds_dict[descriptor][f"{sim_type}_profile"]
        opt_tor = self.ds_dict[descriptor][f"{sim_type}_mean_opt_tor"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        delta = self.ds_dict[descriptor][f"{sim_type}_mean_delta_theta_m"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        actual = self.ds_dict[descriptor]["opt_tor"].loc[
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
                alpha=0.5,
                color=color,
                label="profile",
            )
            ax.plot(
                toroidal_range,
                gaussian_profile,
                alpha=0.5,
                color="r",
                linestyle="--",
                label="gaussian fit",
            )
            ax.axvline(
                opt_tor_val,
                label="mean gaussian-fitted peak",
                color="r",
                linestyle="--",
            )
            ax.axvline(
                actual.loc[{"frequency": frequency}],
                label="predicted opt_tor",
                color=color,
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
        plt.xlabel("toroidal steering/deg")
        plt.ylabel("normalized received power")
        return fig, axes

    def plot_opt_tor_hist(self, descriptor, sim_type="mismatch"):
        """_summary_

        Args:
            descriptor (_type_): _description_
            sim_type (str, optional): One of ('mismatch', 'loc_m', 'loc_product'). Defaults to 'mismatch'.

        Returns:
            _type_: _description_
        """
        mean = self.ds_dict[descriptor][f"{sim_type}_mean_opt_tor"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        actual = self.ds_dict[descriptor]["opt_tor"].loc[
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
                mean.loc[{"frequency": frequency}],
                label="mean",
                color="r",
                linestyle="--",
            )
            ax.axvline(
                actual.loc[{"frequency": frequency}], label="predicted", color="k"
            )
            ax.set_title(f"{frequency} GHz", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=5)

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
        plt.xlabel("Measured optimum toroidal steering/deg")
        plt.ylabel("Number of measurements")

        return fig, axes

    def plot_variable_vs_tor(self, variable, descriptor):
        data = self.ds_dict[descriptor][variable]
        actual = self.ds_dict[descriptor]["opt_tor"].loc[
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

            ax.plot(
                self.toroidal_angles,
                data.loc[
                    {"frequency": frequency, "poloidal_angle": self.poloidal_angles[0]}
                ],
                label=f"{variable}",
            )
            ax.axvline(actual.loc[{"frequency": frequency}], label="opt_tor", color="k")
            ax.set_title(f"{frequency} GHz", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=5)

            counter += 1

        ax.legend(
            bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0.0, fontsize=8
        )
        fig.suptitle(f"{variable} vs. Toroidal Steering", fontsize=15)
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
        plt.xlabel("toroidal steering/deg")
        plt.ylabel(f"{variable}")
        return fig, axes

    def plot_all_opt_tor_measurements(self, sim_type="mismatch"):
        fig = plt.figure()
        plt.title(f"{sim_type}, Opt Tor vs. Probe Frequency")
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
            actual = self.ds_dict[descriptor]["opt_tor"].loc[
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
        plt.ylabel("Opt. Toroidal Angle/Deg")
        plt.legend(
            custom_handles,
            custom_labels,
            bbox_to_anchor=(1.05, 0),
            loc="lower left",
            borderaxespad=0.0,
        )
        return fig

    def plot_opt_tor_vs_freq(self, descriptor, sim_type="mismatch"):
        mean = self.ds_dict[descriptor][f"{sim_type}_mean_opt_tor"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        std = self.ds_dict[descriptor][f"{sim_type}_std_opt_tor"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        actual = self.ds_dict[descriptor]["opt_tor"].loc[
            {"poloidal_angle": self.poloidal_angles[0]}
        ]
        x_coords = self.frequencies
        fig = plt.figure()
        plt.title(f"{sim_type}, Opt Tor vs. Probe Frequency")
        plt.plot(x_coords, mean, label="mean", marker="+")
        plt.fill_between(x_coords, mean - std, mean + std, label="1$\sigma$", alpha=0.3)
        plt.plot(x_coords, actual, label="predicted", color="r", marker="+")
        plt.xlabel("Frequency/GHz")
        plt.ylabel("Opt. Toroidal Angle/Deg")
        plt.legend()
        return fig

    def plot_opt_tor_vs_rho(
        self,
    ):  # TODO: Plot against rho; need to separate rho-freq relation based
        # on theoretical equilibrium
        return

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
        plt.title("rho(f) and rho shift over toroidal sweep range")
        plt.legend(bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0.0)
        return fig
