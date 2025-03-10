# to implement functions to
# - convert rho, theta, zeta <=> x, y, z --- DONE

from desc.equilibrium.coords import map_coordinates
import numpy as np



# for coordinate convention, see section on "Flux coordinates"
# https://desc-docs.readthedocs.io/en/latest/theory_general.html
def rtz2xyz(equilibrium, rho, theta, zeta):
    RphiZ, _ = map_coordinates(equilibrium,
                               np.array([[rho, theta, zeta], [rho, theta, zeta]]),
                               inbasis=("rho","theta","zeta"),
                               outbasis=("R","phi","Z"))
    R, phi, Z = RphiZ[0], RphiZ[1], RphiZ[2]
    X, Y = R*np.cos(phi), R*np.sin(phi)
    return (X, Y, Z)

# NOTE: Sometimes, the value of theta returned may not be between 0 and 2pi.
def xyz2rtz(equilibrium, X, Y, Z):
    R, phi = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)
    rtz, _ = map_coordinates(equilibrium,
                             np.array([[R, phi, Z], [R, phi, Z]]),
                             inbasis=("R","phi","Z"),
                             outbasis=("rho","theta","zeta"))
    rho, theta, zeta = rtz[0], np.remainder(rtz[1], 2*np.pi), np.remainder(rtz[2], 2*np.pi)
    return (rho, theta, zeta)