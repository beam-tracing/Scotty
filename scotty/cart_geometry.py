# Copyright 2023 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

from __future__ import annotations

from abc import ABC
import pathlib
from typing import Callable, Optional, Tuple

from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

from scotty.derivatives import derivative
from scotty.fun_general import find_nearest
from scotty.typing import ArrayLike, FloatArray


class CartMagneticField(ABC):
    """Abstract base class for magnetic field geometries"""

    #: Sample locations for the X coordinate
    X_coord: FloatArray
    #: Sample locations for the Y coordinate
    Y_coord: FloatArray
    #: Sample locations for the Z coordinate
    Z_coord: FloatArray
    #: Value of the poloidal magnetic flux, :math:`\psi`, on ``(R_coord, Z_coord)``
    poloidalFlux_grid: FloatArray

    def B_X(self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def B_Y(self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def B_Z(self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def poloidal_flux(
        self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike
    ) -> FloatArray:
        raise NotImplementedError

    def d_poloidal_flux_dX(
        self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike, delta_X: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            "q_X",
            {"q_X": q_X, "q_Y": q_Y, "q_Z": q_Z},
            {"q_X": delta_X},
        )

    def d_poloidal_flux_dY(
        self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike, delta_Y: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            "q_Y",
            {"q_X": q_X, "q_Y": q_Y, "q_Z": q_Z},
            {"q_Y": delta_Y},
        )

    def d_poloidal_flux_dZ(
        self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike, delta_Z: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            "q_Z",
            {"q_X": q_X, "q_Y": q_Y, "q_Z": q_Z},
            {"q_Z": delta_Z},
        )

    def d2_poloidal_flux_dX2(
        self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike, delta_X: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            ("q_X", "q_X"),
            {"q_X": q_X, "q_Y": q_Y, "q_Z": q_Z},
            {"q_X": delta_X},
        )

    def d2_poloidal_flux_dY2(
        self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike, delta_Y: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            ("q_Y", "q_Y"),
            {"q_X": q_X, "q_Y": q_Y, "q_Z": q_Z},
            {"q_Y": delta_Y},
        )

    def d2_poloidal_flux_dZ2(
        self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike, delta_Z: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            ("q_Z", "q_Z"),
            {"q_X": q_X, "q_Y": q_Y, "q_Z": q_Z},
            {"q_Z": delta_Z},
        )

    def d2_poloidal_flux_dYdZ(
        self,
        q_X: ArrayLike,
        q_Y: ArrayLike,
        q_Z: ArrayLike,
        delta_Y: float,
        delta_Z: float,
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            ("q_Y", "q_Z"),
            {"q_X": q_X, "q_Y": q_Y, "q_Z": q_Z},
            {"q_Y": delta_Y, "q_Z": delta_Z},
        )

    def d2_poloidal_flux_dXdZ(
        self,
        q_X: ArrayLike,
        q_Y: ArrayLike,
        q_Z: ArrayLike,
        delta_X: float,
        delta_Z: float,
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            ("q_X", "q_Z"),
            {"q_X": q_X, "q_Y": q_Y, "q_Z": q_Z},
            {"q_X": delta_X, "q_Z": delta_Z},
        )

    def d2_poloidal_flux_dXdY(
        self,
        q_X: ArrayLike,
        q_Y: ArrayLike,
        q_Z: ArrayLike,
        delta_X: float,
        delta_Y: float,
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            ("q_X", "q_Y"),
            {"q_X": q_X, "q_Y": q_Y, "q_Z": q_Z},
            {"q_X": delta_X, "q_Y": delta_Y},
        )


class CartSlabField(CartMagneticField):
    """Cartesian slab geometry"""

    # change the Y to be the B field
    def __init__(
        self,
        B_Y_0: float,
        R_axis: float,
        minor_radius_a: float,
        grid_buffer_factor: float,
    ):
        # self.B_T_axis = B_T_axis
        self.R_axis = R_axis
        self.minor_radius_a = minor_radius_a
        grid_width = grid_buffer_factor * minor_radius_a

        self.X_coord = np.linspace(R_axis - grid_width, R_axis + grid_width, 101)
        self.Y_coord = np.linspace(-grid_width, grid_width, 101)
        self.Z_coord = np.linspace(-grid_width, grid_width, 101)
        grid_buffer_factor: float = (1,)
        B_Y_0 = 1.0  # in teslas

    def B_X(self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_X, q_Y, q_Z = np.asfarray(q_X), np.asfarray(q_Y), np.asfarray(q_Z)
        return np.zeros_like(q_X)

    def B_Y(self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_X, q_Y, q_Z = np.asfarray(q_X), np.asfarray(q_Y), np.asfarray(q_Z)
        return np.ones_like(q_Y) * B_Y_0
        # B_Y is constant

    def B_Z(self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_X, q_Y, q_Z = np.asfarray(q_X), np.asfarray(q_Y), np.asfarray(q_Z)
        return np.zeros_like(q_Z)

    def poloidal_flux(
        self, q_X: ArrayLike, q_Y: ArrayLike, q_Z: ArrayLike
    ) -> FloatArray:
        return 1 / minor_radius_a
        # linear in x
