"""
Read and write TORBEAM files
"""
from __future__ import annotations

from scotty.fun_general import read_floats_into_list_until
from scotty.typing import PathLike

import freegs._fileutils as fortran


class Torbeam:
    """Utility class for reading/writing TORBEAM geometry files

    Arguments
    ---------
    R_grid:
        Major radius grid points in metres
    Z_grid:
        Vertical coordinate grid points in metres
    B_R:
        Major-radial magnetic field component in Tesla
    B_T:
        Toroidal magnetic field component in Tesla
    B_Z:
        Vertical magnetic field component in Tesla
    psi:
        Poloidal flux
    """

    def __init__(self, R_grid, Z_grid, B_R, B_T, B_Z, psi):
        self.R_grid = R_grid
        self.Z_grid = Z_grid
        self.B_R = B_R
        self.B_T = B_T
        self.B_Z = B_Z
        self.psi = psi

    @classmethod
    def from_file(cls, filename: PathLike) -> Torbeam:
        """Read TORBEAM geometry file"""

        with open(filename) as f:
            while "X-coordinates" not in f.readline():
                pass  # Start reading only from X-coords onwards
            R_grid = read_floats_into_list_until("Z-coordinates", f)
            Z_grid = read_floats_into_list_until("B_R", f)
            R_points = len(R_grid)
            Z_points = len(Z_grid)

            # Row-major and column-major business (Torbeam is in
            # Fortran and Scotty is in Python)
            def from_fortran(array):
                return array.reshape((Z_points, R_points)).transpose()

            B_R = from_fortran(read_floats_into_list_until("B_t", f))
            B_T = from_fortran(read_floats_into_list_until("B_Z", f))
            B_Z = from_fortran(read_floats_into_list_until("psi", f))
            poloidal_flux = from_fortran(
                read_floats_into_list_until("you fall asleep", f)
            )

        return cls(R_grid, Z_grid, B_R, B_T, B_Z, poloidal_flux)

    def write(self, filename: PathLike):
        """Write TORBEAM geometry file"""

        x_grid_length = len(self.R_grid)
        z_grid_length = len(self.Z_grid)

        with open(filename, "w") as topfile_file:
            chunks = fortran.ChunkOutput(topfile_file, extraspaces=1)

            topfile_file.write("Dummy line\n")
            topfile_file.write(f"{x_grid_length} {z_grid_length}\n")
            topfile_file.write("Dummy line\n")
            topfile_file.write("0 0 1\n")
            topfile_file.write("Grid: X-coordinates\n")
            for ii in range(x_grid_length):
                topfile_file.write("{:.8e}\n".format(self.R_grid[ii]))

            topfile_file.write("Grid: Z-coordinates\n")
            fortran.write_1d(self.Z_grid, chunks)

            topfile_file.write("Magnetic field: B_R\n")
            fortran.write_2d(self.B_R, chunks)

            topfile_file.write("Magnetic field: B_t\n")
            fortran.write_2d(self.B_T, chunks)

            topfile_file.write("Magnetic field: B_Z\n")
            fortran.write_2d(self.B_Z, chunks)

            topfile_file.write("Poloidal flux: psi\n")
            fortran.write_2d(self.psi, chunks)
