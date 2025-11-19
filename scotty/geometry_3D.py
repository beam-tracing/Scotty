from abc import ABC
import json
import logging
import numpy as np
import pathlib
from scipy.interpolate import RegularGridInterpolator
from scotty.derivatives import derivative
from scotty.logger_3D import timer
from scotty.typing import ArrayLike, FloatArray
from typing import Callable, Tuple, Optional, Union

log = logging.getLogger(__name__)

##################################################
#
# FIELD INTERPOLATION CODES
#
##################################################

# Implementing an interpolation method for $B_X$, $B_Y$, $B_Z$, and $polflux$.
# *interp_smoothing no longer supported.

def _interp_mesh3D_data1D(
        X_grid, Y_grid, Z_grid,
        data_array,
        interp_order):
# ) -> Tuple[Callable[[ArrayLike, ArrayLike, ArrayLike], [FloatArray, FloatArray, FloatArray]]]:

    spline = RegularGridInterpolator(
        points = (X_grid, Y_grid, Z_grid),
        values = data_array,
        method = interp_order,
        bounds_error = False,
    )

    return lambda X,Y,Z: spline((X,Y,Z)), spline



# ABC is an abstract base class, so we implement methods to force child classes to implement these methods as well.
class MagneticField_3D_Cartesian(ABC):
    X_coord: FloatArray
    Y_coord: FloatArray
    Z_coord: FloatArray
    # polflux_grid: FloatArray # TO REMOVE -- put this back/remove when the other polflux_grid below is back/removed also

    # Declaring abstract methods (child classes must implement this method)
    # for B_X, B_Y, B_Z, and polflux
    def B_X(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def B_Y(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def B_Z(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def polflux(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    
    # Declaring abstract methods (child classes must implement this method)
    # for the first-order and second-order derivatives of polflux
    def d_polflux_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float) -> FloatArray: raise NotImplementedError
    def d_polflux_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float) -> FloatArray: raise NotImplementedError
    def d_polflux_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dX2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dY2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dZ2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dXdY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Y: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dXdZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Z: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dYdZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float, delta_Z: float) -> FloatArray: raise NotImplementedError
    
    # Declaring abstract methods (child classes must implement this method)
    # for the magnitude and the unit vector of the magnetic field
    def magnitude(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return np.sqrt( self.B_X(X,Y,Z)**2 + self.B_Y(X,Y,Z)**2 + self.B_Z(X,Y,Z)**2 )
    
    def unitvector(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        magnitude = self.magnitude(X,Y,Z)
        vector = np.array( [self.B_X(X,Y,Z), self.B_Y(X,Y,Z), self.B_Z(X,Y,Z)] )
        return (vector / magnitude).T



# Implementing a child class of the abstract base class MagneticField_3D_Cartesian.
class InterpolatedField_3D_Cartesian(MagneticField_3D_Cartesian):

    def __init__(self, X_grid: FloatArray, Y_grid: FloatArray, Z_grid: FloatArray,
                 B_X: FloatArray, B_Y: FloatArray, B_Z: FloatArray, psi: FloatArray,
                 interp_order: int = 5):
        
        self.X_coord = X_grid
        self.Y_coord = Y_grid
        self.Z_coord = Z_grid
        # self.polflux_grid = psi # TO REMOVE -- do I need to put this back? a lot of memory used for no reason?

        ((self._interp_B_X,
          self.spline_B_X),
          duration_B_X_interpolation) = timer(_interp_mesh3D_data1D, X_grid, Y_grid, Z_grid, B_X, interp_order)
        log.debug(f"Interpolating 3D B_X profile took {duration_B_X_interpolation} s")
        
        ((self._interp_B_Y,
          self.spline_B_Y),
          duration_B_Y_interpolation) = timer(_interp_mesh3D_data1D, X_grid, Y_grid, Z_grid, B_Y, interp_order)
        log.debug(f"Interpolating 3D B_Y profile took {duration_B_Y_interpolation} s")
        
        ((self._interp_B_Z,
          self.spline_B_Z),
          duration_B_Z_interpolation) = timer(_interp_mesh3D_data1D, X_grid, Y_grid, Z_grid, B_Z, interp_order)
        log.debug(f"Interpolating 3D B_Z profile took {duration_B_Z_interpolation} s")
        
        ((self._interp_polflux,
          self.spline_polflux),
          duration_polflux_interpolation) = timer(_interp_mesh3D_data1D, X_grid, Y_grid, Z_grid, psi, interp_order)
        log.debug(f"Interpolating 3D poloidal flux profile took {duration_polflux_interpolation} s")
    
    # Defining the class attributes for B_X, B_Y, B_Z, and polflux
    def B_X(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_B_X(X,Y,Z)
    def B_Y(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_B_Y(X,Y,Z)
    def B_Z(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_B_Z(X,Y,Z)
    def polflux(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_polflux(X,Y,Z)
    
    # TO REMOVE -- 30 August
    # Defining class attributes for first order derivatives of B field -- for future work?
    # def d_B_X_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.spline_B_X((X,Y,Z), nu=[1,0,0])
    # def d_B_X_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.spline_B_X((X,Y,Z), nu=[0,1,0])
    # def d_B_X_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.spline_B_X((X,Y,Z), nu=[0,0,1])

    # def d_B_Y_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.spline_B_Y((X,Y,Z), nu=[1,0,0])
    # def d_B_Y_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.spline_B_Y((X,Y,Z), nu=[0,1,0])
    # def d_B_Y_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.spline_B_Y((X,Y,Z), nu=[0,0,1])

    # def d_B_Z_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.spline_B_Z((X,Y,Z), nu=[1,0,0])
    # def d_B_Z_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.spline_B_Z((X,Y,Z), nu=[0,1,0])
    # def d_B_Z_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.spline_B_Z((X,Y,Z), nu=[0,0,1])

    # Defining the class attributes for the first-order and second-order derivatives of polflux
    def d_polflux_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float) -> FloatArray:
        return derivative(self.polflux, ("X"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X})
    
    def d_polflux_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float) -> FloatArray:
        return derivative(self.polflux, ("Y"), {"X": X, "Y": Y, "Z": Z}, {"Y": delta_Y})
    
    def d_polflux_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float) -> FloatArray:
        return derivative(self.polflux, ("Z"), {"X": X, "Y": Y, "Z": Z}, {"Z": delta_Z})
    
    def d2_polflux_dX2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float) -> FloatArray:
        return derivative(self.polflux, ("X", "X"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X})
    
    def d2_polflux_dY2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float) -> FloatArray:
        return derivative(self.polflux, ("Y", "Y"), {"X": X, "Y": Y, "Z": Z}, {"Y": delta_Y})
    
    def d2_polflux_dZ2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float) -> FloatArray:
        return derivative(self.polflux, ("Z", "Z"), {"X": X, "Y": Y, "Z": Z}, {"Z": delta_Z})
    
    def d2_polflux_dXdY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Y: float) -> FloatArray:
        return derivative(self.polflux, ("X", "Y"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X, "Y": delta_Y})
    
    def d2_polflux_dXdZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Z: float) -> FloatArray:
        return derivative(self.polflux, ("X", "Z"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X, "Z": delta_Z})
    
    def d2_polflux_dYdZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float, delta_Z: float) -> FloatArray:
        return derivative(self.polflux, ("Y", "Z"), {"X": X, "Y": Y, "Z": Z}, {"Y": delta_Y, "Z": delta_Z})



def create_magnetic_geometry_3D(
    find_B_method: Union[str, MagneticField_3D_Cartesian],
    magnetic_data_path: Optional[pathlib.Path] = None,
    input_filename_suffix: str = "",
    interp_order: int = 5,
    shot: Optional[int] = None,
    equil_time: Optional[float] = None,
    **kwargs: dict) -> MagneticField_3D_Cartesian:

    log.debug(f"Reading and creating field profile")
    
    if isinstance(find_B_method, MagneticField_3D_Cartesian):
        log.debug(f"Using existing field profile passed from `find_B_method`")
        return find_B_method

    def missing_arg(argument: str) -> str:
        return f"Missing '{argument}' for find_B_method='{find_B_method}'"
    
    if find_B_method == "omfit_3D":
        raise ValueError(f"`omfit_3D` in `find_B_method` was renamed to `eduard_3D`!")
    
    if find_B_method == "eduard_3D":
        topfile_filename = magnetic_data_path / f"topfile{input_filename_suffix}.json"
        log.info(f"Using a custom file for B and polflux from {topfile_filename}")

        with open(topfile_filename) as f: data = json.load(f)
        X_grid = np.array(data["X"])
        Y_grid = np.array(data["Y"])
        Z_grid = np.array(data["Z"])
        log.debug(f"Loaded X, Y, Z grids with shape {X_grid.shape}, {Y_grid.shape}, {Z_grid.shape}")

        try:             B_X = np.array(data["Bx"])
        except KeyError: B_X = np.array(data["B_X"])
        except Exception as e: raise e(f"{e}: Failed to read B_X data")
        else: log.debug(f"Loaded B_X data with shape {B_X.shape}")

        try:             B_Y = np.array(data["By"])
        except KeyError: B_Y = np.array(data["B_Y"])
        except Exception as e: raise e(f"{e}: Failed to read B_Y data")
        else: log.debug(f"Loaded B_Y data with shape {B_Y.shape}")

        try:             B_Z = np.array(data["Bz"])
        except KeyError: B_Z = np.array(data["B_Z"])
        except Exception as e: raise e(f"{e}: Failed to read B_Z data")
        else: log.debug(f"Loaded B_Z data with shape {B_Z.shape}")
        
        try:             polflux = np.array(data["pol_flux"])
        except KeyError: polflux = np.array(data["polflux"])
        except Exception as e: raise e(f"{e}: Failed to read poloidal flux data")
        else: log.debug(f"Loaded poloidal_flux data with shape {polflux.shape}")

        (field,
         duration_field_interpolation) = timer(InterpolatedField_3D_Cartesian,
                                               X_grid, Y_grid, Z_grid,
                                               B_X, B_Y, B_Z, polflux,
                                               interp_order)
        
        log.debug(f"Reading and creating field profile took {duration_field_interpolation} s")

        return field










# TO REMOVE?
# Testing local grid interpolation instead of entire-field interpolations
# Incomplete

def find_local_interp_cube(coord, grid_coord, N_cube=6):
    left_bound  = np.searchsorted(grid_coord, coord) - 1 - N_cube//2
    right_bound = np.searchsorted(grid_coord, coord) + 1 + N_cube//2
    return grid_coord[left_bound:right_bound], left_bound, right_bound

def make_local_interp_cube(q_X, q_Y, q_Z, grid_X, grid_Y, grid_Z, grid_data):
    local_X_grid, x_lb, x_rb = find_local_interp_cube(q_X, grid_X)
    local_Y_grid, y_lb, y_rb = find_local_interp_cube(q_Y, grid_Y)
    local_Z_grid, z_lb, z_rb = find_local_interp_cube(q_Z, grid_Z)
    local_data_grid = grid_data[x_lb:x_rb, y_lb:y_rb, z_lb:z_rb]

    spline = RegularGridInterpolator(
        points = (local_X_grid, local_Y_grid, local_Z_grid),
        values = local_data_grid,
        method = "quintic",
        bounds_error = False,
    )

    return lambda X,Y,Z: spline((X,Y,Z)), spline