### Declaring class MagneticField_3D_Cartesian(ABC)
# ABC is an abstract base class, so we implement methods to force child classes to implement these methods as well.

from abc import ABC
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scotty.derivatives
from scotty.typing import ArrayLike, FloatArray
from typing import Callable, Tuple

class MagneticField_3D_Cartesian(ABC):
    X_coord: FloatArray
    Y_coord: FloatArray
    Z_coord: FloatArray
    polflux_grid: FloatArray

    # Declaring abstract methods (child classes must implement this method)
    # Specifically for B_X, B_Y, B_Z, and polflux

    def B_X(self,
            X: ArrayLike,
            Y: ArrayLike,
            Z: ArrayLike) -> FloatArray:
        raise NotImplementedError
    
    def B_Y(self,
            X: ArrayLike,
            Y: ArrayLike,
            Z: ArrayLike) -> FloatArray:
        raise NotImplementedError
    
    def B_Z(self,
            X: ArrayLike,
            Y: ArrayLike,
            Z: ArrayLike) -> FloatArray:
        raise NotImplementedError
    
    def polflux(self,
            X: ArrayLike,
            Y: ArrayLike,
            Z: ArrayLike) -> FloatArray:
        raise NotImplementedError
    
    # Declaring abstract methods (child classes must implement this method)
    # Specifically for the first-order and second-order derivatives of polflux

    def d_polflux_dX(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_X: float) -> FloatArray:
        raise NotImplementedError
    
    def d_polflux_dY(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Y: float) -> FloatArray:
        raise NotImplementedError
    
    def d_polflux_dZ(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Z: float) -> FloatArray:
        raise NotImplementedError
    
    def d2_polflux_dX2(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_X: float) -> FloatArray:
        raise NotImplementedError
    
    def d2_polflux_dY2(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Y: float) -> FloatArray:
        raise NotImplementedError
    
    def d2_polflux_dZ2(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Z: float) -> FloatArray:
        raise NotImplementedError
    
    def d2_polflux_dXdY(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_X: float,
                     delta_Y: float) -> FloatArray:
        raise NotImplementedError
    
    def d2_polflux_dXdZ(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_X: float,
                     delta_Y: float) -> FloatArray:
        raise NotImplementedError
    
    def d2_polflux_dYdZ(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Y: float,
                     delta_Z: float) -> FloatArray:
        raise NotImplementedError
    
    # Declaring abstract methods (child classes must implement this method)
    # Specifically for the magnitude and the unit vector of the magnetic field
    #   at some specific point(s) in space
    
    def magnitude(self,
                  X: ArrayLike,
                  Y: ArrayLike,
                  Z: ArrayLike) -> FloatArray:
        return np.sqrt(self.B_X(X,Y,Z)**2 + self.B_Y(X,Y,Z)**2 + self.B_Z(X,Y,Z)**2)
    
    def unitvector(self,
                   X: ArrayLike,
                   Y: ArrayLike,
                   Z: ArrayLike) -> FloatArray:
        magnitude = self.magnitude(X,Y,Z)
        unit_vector = np.array([self.B_X(X,Y,Z), self.B_Y(X,Y,Z), self.B_Z(X,Y,Z)])
        return (unit_vector / magnitude).T





### Declaring class InterpolatedField_3D_Cartesian(MagneticField_3D_Cartesian)
# Implementing a child class of the abstract base class MagneticField_3D_Cartesian.

class InterpolatedField_3D_Cartesian(MagneticField_3D_Cartesian):

    def __init__(self,
                 X_grid: FloatArray,
                 Y_grid: FloatArray,
                 Z_grid: FloatArray,
                 B_X: FloatArray,
                 B_Y: FloatArray,
                 B_Z: FloatArray,
                 psi: FloatArray,
                 interp_order: int = 5):
        
        self.X_coord = X_grid
        self.Y_coord = Y_grid
        self.Z_coord = Z_grid
        self.polflux_grid = psi
        
        self._interp_B_X, _ = _interp_mesh3D_data1D(
            X_grid, Y_grid, Z_grid, B_X, interp_order)
        
        self._interp_B_Y, _ = _interp_mesh3D_data1D(
            X_grid, Y_grid, Z_grid, B_Y, interp_order)
        
        self._interp_B_Z, _ = _interp_mesh3D_data1D(
            X_grid, Y_grid, Z_grid, B_Z, interp_order)
        
        self._interp_polflux, spline_polflux = _interp_mesh3D_data1D(
            X_grid, Y_grid, Z_grid, psi, interp_order)
    
    # Defining the class attributes
    # Specifically for B_X, B_Y, B_Z, and polflux

    def B_X(self,
            X: ArrayLike,
            Y: ArrayLike,
            Z: ArrayLike) -> FloatArray:
        return self._interp_B_X(X,Y,Z)
    
    def B_Y(self,
            X: ArrayLike,
            Y: ArrayLike,
            Z: ArrayLike) -> FloatArray:
        return self._interp_B_Y(X,Y,Z)
    
    def B_Z(self,
            X: ArrayLike,
            Y: ArrayLike,
            Z: ArrayLike) -> FloatArray:
        return self._interp_B_Z(X,Y,Z)
    
    def polflux(self,
            X: ArrayLike,
            Y: ArrayLike,
            Z: ArrayLike) -> FloatArray:
        return self._interp_polflux(X,Y,Z)

    # Defining the class attributes
    # Specifically for the first-order and second-order derivatives of polflux

    def d_polflux_dX(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_X: float) -> FloatArray:
        return scotty.derivatives.derivative(self.polflux,
                                             ("X"),
                                             {"X":X, "Y":Y, "Z":Z},
                                             {"X":delta_X})
    
    def d_polflux_dY(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Y: float) -> FloatArray:
        return scotty.derivatives.derivative(self.polflux,
                                             ("Y"),
                                             {"X":X, "Y":Y, "Z":Z},
                                             {"Y":delta_Y})
    
    def d_polflux_dZ(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Z: float) -> FloatArray:
        return scotty.derivatives.derivative(self.polflux,
                                             ("Z"),
                                             {"X":X, "Y":Y, "Z":Z},
                                             {"Z":delta_Z})
    
    def d2_polflux_dX2(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_X: float) -> FloatArray:
        return scotty.derivatives.derivative(self.polflux,
                                             ("X","X"),
                                             {"X":X, "Y":Y, "Z":Z},
                                             {"X":delta_X})
    
    def d2_polflux_dY2(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Y: float) -> FloatArray:
        return scotty.derivatives.derivative(self.polflux,
                                             ("Y","Y"),
                                             {"X":X, "Y":Y, "Z":Z},
                                             {"Y":delta_Y})
    
    def d2_polflux_dZ2(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Z: float) -> FloatArray:
        return scotty.derivatives.derivative(self.polflux,
                                             ("Z","Z"),
                                             {"X":X, "Y":Y, "Z":Z},
                                             {"Z":delta_Z})
    
    def d2_polflux_dXdY(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_X: float,
                     delta_Y: float) -> FloatArray:
        return scotty.derivatives.derivative(self.polflux,
                                             ("X","Y"),
                                             {"X":X, "Y":Y, "Z":Z},
                                             {"X":delta_X, "Y":delta_Y})
    
    def d2_polflux_dXdZ(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_X: float,
                     delta_Z: float) -> FloatArray:
        return scotty.derivatives.derivative(self.polflux,
                                             ("X","Z"),
                                             {"X":X, "Y":Y, "Z":Z},
                                             {"X":delta_X, "Z":delta_Z})
    
    def d2_polflux_dYdZ(self,
                     X: ArrayLike,
                     Y: ArrayLike,
                     Z: ArrayLike,
                     delta_Y: float,
                     delta_Z: float) -> FloatArray:
        return scotty.derivatives.derivative(self.polflux,
                                             ("Y","Z"),
                                             {"X":X, "Y":Y, "Z":Z},
                                             {"Y":delta_Z, "Z":delta_Z})





### Declaring def _interp_mesh3D_data1D
# Implementing an interpolation method for $B_X$, $B_Y$, $B_Z$, and $polflux$.
# *interp_smoothing no longer supported.

def _interp_mesh3D_data1D(
        X_grid, Y_grid, Z_grid,
        data_array,
        interp_order: int,
) -> Tuple[Callable[[ArrayLike, ArrayLike, ArrayLike], [FloatArray, FloatArray, FloatArray]]]:
    
    if   interp_order == 1: interp_order_str = "linear"
    elif interp_order == 3: interp_order_str = "cubic"
    elif interp_order == 5: interp_order_str = "quintic"
    else:
        interp_order_str = "quintic"
        print("interp_order not supported. Setting interp_order = 5 for _interp_mesh3D_data1D")

    spline = RegularGridInterpolator(
        points = (X_grid, Y_grid, Z_grid),
        values = data_array,
        method = interp_order_str,
        bounds_error = False,
    )

    return lambda X,Y,Z: spline((X,Y,Z)), spline