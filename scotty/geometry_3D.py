from abc import ABC
import json
import logging
import numpy as np
import pathlib
from scipy.interpolate import RegularGridInterpolator
from scotty.derivatives import derivative
from scotty.geometry import MagneticField
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
        X_coord, Y_coord, Z_coord,
        data_array,
        interp_order):
# ) -> Tuple[Callable[[ArrayLike, ArrayLike, ArrayLike], [FloatArray, FloatArray, FloatArray]]]:

    spline = RegularGridInterpolator(
        points = (X_coord, Y_coord, Z_coord),
        values = data_array,
        method = interp_order,
        bounds_error = False,
    )

    return lambda X,Y,Z: spline((X,Y,Z)), spline

##################################################
#
# FIELD CLASSES
#
##################################################

# ABC is an abstract base class, so we implement methods to force child classes to implement these methods as well.
class MagneticField_Cartesian(ABC):
    X_coord: FloatArray
    Y_coord: FloatArray
    Z_coord: FloatArray

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



# Implementing a child class of the abstract base class MagneticField_Cartesian.
class InterpolatedField_Cartesian(MagneticField_Cartesian):

    r"""Interpolated numerical equilibrium computed using SciPy RegularGridInterpolator
    
    Parameters
    ----------
    X_coord : ArrayLike, shape (n1,)
        1D array of points in ``X`` (metres)
    
    Y_coord : ArrayLike, shape (n2,)
        1D array of points in ``Y`` (metres)
    
    Z_coord : ArrayLike, shape (n3,)
        1D array of points in ``Z`` (metres)
    
    B_X : ArrayLike, shape (n1, n2, n3)
        3D ``(X, Y, Z)`` grid of the X-component magnetic field
    
    B_Y : ArrayLike, shape (n1, n2, n3)
        3D ``(X, Y, Z)`` grid of the Y-component magnetic field
    
    B_Z : ArrayLike, shape (n1, n2, n3)
        3D ``(X, Y, Z)`` grid of the Z-component magnetic field
    
    psi : ArrayLike, shape (n1, n2, n3)
        3D ``(X, Y, Z)`` grid of the normalised poloidal flux values
    
    interp_order : int
        Order of interpolating splines. Default is 5 (quintic)
    
    Attributes
    ----------
    X_coord : ArrayLike, shape (n1,)
        1D array of points in ``X`` (metres)
    
    Y_coord : ArrayLike, shape (n2,)
        1D array of points in ``Y`` (metres)
    
    Z_coord : ArrayLike, shape (n3,)
        1D array of points in ``Z`` (metres)
    
    grid_coords : ArrayLike, shape (n1, n2, n3)
        3D ``(X, Y, Z)`` grid equivalent to ``np.meshgrid(X_coord, Y_coord, Z_coord, indexing='ij')``
    
    B_X : ArrayLike, shape (n1, n2, n3)
        3D ``(X, Y, Z)`` grid of the X-component magnetic field values, in units of Tesla
    
    B_Y : ArrayLike, shape (n1, n2, n3)
        3D ``(X, Y, Z)`` grid of the Y-component magnetic field values, in units of Tesla
    
    B_Z : ArrayLike, shape (n1, n2, n3)
        3D ``(X, Y, Z)`` grid of the Z-component magnetic field values, in units of Tesla
    
    psi : ArrayLike, shape (n1, n2, n3)
        3D ``(X, Y, Z)`` grid of the normalised poloidal flux values
    
    interp_order : int, optional
        Order of interpolating splines. Default is ``5`` (quintic).
    
    Methods
    ----------
    B_X : Interpolated spline of X-component magnetic field values at coordinates

    B_Y : Interpolated spline of Y-component magnetic field values at coordinates

    B_Z : Interpolated spline of Z-component magnetic field values at coordinates

    polflux : Interpolated spline of normalised poloidal flux value at coordinates

    d_polflux_dX : Numerically-calculated derivative at coordinates, first-order in X

    d_polflux_dY : Numerically-calculated derivative at coordinates, first-order in Y

    d_polflux_dZ : Numerically-calculated derivative at coordinates, first-order in Z

    d2_polflux_dX2 : Numerically-calculated derivative at coordinates, second-order in X

    d2_polflux_dY2 : Numerically-calculated derivative at coordinates, second-order in Y

    d2_polflux_dZ2 : Numerically-calculated derivative at coordinates, second-order in Z

    d2_polflux_dXdY : Numerically-calculated derivative at coordinates, second-order in X and Y
    
    d2_polflux_dXdZ : Numerically-calculated derivative at coordinates, second-order in X and Z

    d2_polflux_dYdZ : Numerically-calculated derivative at coordinates, second-order in Y and Z

    """

    def __init__(
        self,
        X_coord: FloatArray,
        Y_coord: FloatArray,
        Z_coord: FloatArray,
        B_X: FloatArray,
        B_Y: FloatArray,
        B_Z: FloatArray,
        psi: FloatArray,
        interp_order: int = 5):
        
        # Defining class attributes for the interpolated grids
        self.X_coord = X_coord
        self.Y_coord = Y_coord
        self.Z_coord = Z_coord
        self.B_X = B_X
        self.B_Y = B_Y
        self.B_Z = B_Z
        self.psi = psi
        self.interp_order = interp_order

        ((self._interp_B_X,
          self._spline_B_X),
          duration_B_X_interpolation) = timer(_interp_mesh3D_data1D, X_coord, Y_coord, Z_coord, B_X, interp_order)
        log.debug(f"Interpolating 3D B_X profile took {duration_B_X_interpolation} s")
        
        ((self._interp_B_Y,
          self._spline_B_Y),
          duration_B_Y_interpolation) = timer(_interp_mesh3D_data1D, X_coord, Y_coord, Z_coord, B_Y, interp_order)
        log.debug(f"Interpolating 3D B_Y profile took {duration_B_Y_interpolation} s")
        
        ((self._interp_B_Z,
          self._spline_B_Z),
          duration_B_Z_interpolation) = timer(_interp_mesh3D_data1D, X_coord, Y_coord, Z_coord, B_Z, interp_order)
        log.debug(f"Interpolating 3D B_Z profile took {duration_B_Z_interpolation} s")
        
        ((self._interp_polflux,
          self._spline_polflux),
          duration_polflux_interpolation) = timer(_interp_mesh3D_data1D, X_coord, Y_coord, Z_coord, psi, interp_order)
        log.debug(f"Interpolating 3D poloidal flux profile took {duration_polflux_interpolation} s")

        self.grid_coords = self._spline_B_X.grid()
    
    # Defining the class attributes for B_X, B_Y, B_Z, and polflux
    def B_X(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_B_X(X,Y,Z)
    def B_Y(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_B_Y(X,Y,Z)
    def B_Z(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_B_Z(X,Y,Z)
    def polflux(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_polflux(X,Y,Z)
    


    # TO REMOVE -- 30 August
    # Defining class attributes for first order derivatives of B field -- for future work?
    # Need to compare with dbhats in analysis.py
    def d_B_X_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_B_X((X,Y,Z), nu=[1,0,0])
    
    def d_B_Y_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_B_Y((X,Y,Z), nu=[1,0,0])
    
    def d_B_Z_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_B_Z((X,Y,Z), nu=[1,0,0])
    
    def d_B_X_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_B_X((X,Y,Z), nu=[0,1,0])
    
    def d_B_Y_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_B_Y((X,Y,Z), nu=[0,1,0])
    
    def d_B_Z_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_B_Z((X,Y,Z), nu=[0,1,0])
    
    def d_B_X_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_B_X((X,Y,Z), nu=[0,0,1])
    
    def d_B_Y_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_B_Y((X,Y,Z), nu=[0,0,1])
    
    def d_B_Z_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_B_Z((X,Y,Z), nu=[0,0,1])
    
    # TO REMOVE: VERY IMPORTANT
    # NEED TO RENAME: this isnt actually d(bhat)/dX, but d(Bvec)/dX
    # Need to use chain rule to get d(bhat)/dX, so might actually be less computationally expensive to use finite differences
    def d_bhat_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return np.array( [self.d_B_X_dX(X,Y,Z), self.d_B_Y_dX(X,Y,Z), self.d_B_Z_dX(X,Y,Z)] ).T
    


    # TO REMOVE
    def d_polflux_dX_TEST(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_polflux((X,Y,Z), nu=[1,0,0])
    def d_polflux_dY_TEST(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_polflux((X,Y,Z), nu=[0,1,0])
    def d_polflux_dZ_TEST(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        return self._spline_polflux((X,Y,Z), nu=[0,0,1])
    



    # Defining the class attributes for the first- and second-order derivatives of polflux
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
    find_B_method, # : Union[str, MagneticField, MagneticField_Cartesian], # TO REMOVE?
    magnetic_data_path: Optional[Union[str, pathlib.Path]] = None,
    input_filename_suffix: str = "",
    interp_order: int = 5,
    shot: Optional[int] = None,
    equil_time: Optional[float] = None,
    **kwargs: Optional[dict]) -> MagneticField_Cartesian:

    r"""Creates a magnetic field equilibrium
    
    Parameters
    ----------
    find_B_method : str or ``MagneticField`` or ``MagneticField_Cartesian`` or str
        Determines how the magnetic field equilibrium data should be read. If
        ``MagneticField`` or ``MagneticField_Cartesian`` is passed, then the data
        has already been interpolated beforehand and no interpolation is done
        internally. If a string is passed, it must be one of "eduard_3D" or "omfit_3D".

    magnetic_data_path : str or ``pathlib.Path``, optional
        Folder directory which contains the field data. Default is ``None``

    input_filename_suffix : str, optional
        Suffix of the field data file. Default is ``""``
    
    interp_order : int, optional
        Order of interpolating splines. Default is ``5`` (quintic)
    
    shot : int, optional
        Shot number of the field data. Default is ``None``

    equil_time : float, optional
        Equilibrium time of the field data. Default is ``None``
    
    Returns
    ----------
    ``MagneticField_Cartesian``
        Interpolated field in 3D ``(X, Y, Z)`` Cartesian coordinates
    
    """

    log.debug(f"Reading and creating field profile")

    if isinstance(magnetic_data_path, str):
        magnetic_data_path = pathlib.Path(magnetic_data_path)

    if isinstance(find_B_method, MagneticField):
        if not kwargs: raise ValueError(f"""
    Extra keyword arguments must be passed into `create_magnetic_geometry_3D` when
    a two-dimensional (R,Z) field profile is passed into `find_B_method`
    """)
        
        log.debug(f"Two-dimensional (R,Z) field profile passed from `find_B_method`")
        coords = {"X": None, "Y": None, "Z": None}
        for dim in coords:
            if dim not in kwargs: raise ValueError(f"Missing keyword argument: {dim}")
            else: coords[dim] = np.array(kwargs[dim])
        
        log.debug(f"Converting the (R,Z)-field into an (X,Y,Z)-field")
        XX, YY, ZZ = np.meshgrid(*[coords[dim] for dim in coords], indexing="ij")
        RR = np.sqrt( XX**2 + YY**2 )

        B_R = find_B_method.B_R(RR, ZZ)
        B_T = find_B_method.B_T(RR, ZZ)
        B_X = (B_R*XX - B_T*YY) / RR
        B_Y = (B_R*YY + B_T*XX) / RR
        B_Z = find_B_method.B_Z(RR, ZZ)
        polflux = find_B_method.poloidal_flux(RR, ZZ)

        (field,
         duration_field_interpolation) = timer(InterpolatedField_Cartesian,
                                               coords["X"], coords["Y"], coords["Z"],
                                               B_X, B_Y, B_Z, polflux,
                                               interp_order)
        
        log.debug(f"Converting the field profile took {duration_field_interpolation} s")

        return field
    
    elif isinstance(find_B_method, MagneticField_Cartesian):
        log.debug(f"Using existing field profile passed from `find_B_method`")
        return find_B_method
    
    elif find_B_method == "eduard_3D":
        topfile_filename = magnetic_data_path / f"topfile{input_filename_suffix}.json"
        log.info(f"Using a custom file for B and polflux from {topfile_filename}")

        with open(topfile_filename) as f: data = json.load(f)
        X_coord = np.array(data["X"])
        Y_coord = np.array(data["Y"])
        Z_coord = np.array(data["Z"])
        log.debug(f"Loaded X, Y, Z grids with shape {X_coord.shape}, {Y_coord.shape}, {Z_coord.shape}")

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
         duration_field_interpolation) = timer(InterpolatedField_Cartesian,
                                               X_coord, Y_coord, Z_coord,
                                               B_X, B_Y, B_Z, polflux,
                                               interp_order)
        
        log.debug(f"Reading and creating field profile took {duration_field_interpolation} s")

        return field
    
    elif find_B_method == "omfit_3D":
        raise ValueError(f"`omfit_3D` in `find_B_method` was renamed to `eduard_3D`!")
    
    else: raise ValueError(f"Unknown `find_B_method` passed!")










# TO REMOVE?
# Testing local grid interpolation instead of entire-field interpolations
# Incomplete

def find_local_interp_cube(coord, grid_coord, N_cube=6):
    left_bound  = np.searchsorted(grid_coord, coord) - 1 - N_cube//2
    right_bound = np.searchsorted(grid_coord, coord) + 1 + N_cube//2
    return grid_coord[left_bound:right_bound], left_bound, right_bound

def make_local_interp_cube(q_X, q_Y, q_Z, grid_X, grid_Y, grid_Z, grid_data):
    local_X_coord, x_lb, x_rb = find_local_interp_cube(q_X, grid_X)
    local_Y_coord, y_lb, y_rb = find_local_interp_cube(q_Y, grid_Y)
    local_Z_coord, z_lb, z_rb = find_local_interp_cube(q_Z, grid_Z)
    local_data_coord = grid_data[x_lb:x_rb, y_lb:y_rb, z_lb:z_rb]

    spline = RegularGridInterpolator(
        points = (local_X_coord, local_Y_coord, local_Z_coord),
        values = local_data_coord,
        method = "quintic",
        bounds_error = False,
    )

    return lambda X,Y,Z: spline((X,Y,Z)), spline