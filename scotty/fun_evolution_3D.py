import logging
import numpy as np
from scipy.integrate import solve_ivp
from scotty.hamiltonian_3D import Hamiltonian_3D, hessians_3D
from scotty.logger_3D import timer
from scotty.typing import FloatArray, ArrayLike
from typing import Tuple

log = logging.getLogger(__name__)

def pack_beam_parameters_3D(
        q_X: ArrayLike,
        q_Y: ArrayLike,
        q_Z: ArrayLike,
        K_X: ArrayLike,
        K_Y: ArrayLike,
        K_Z: ArrayLike,
        Psi: FloatArray,
) -> FloatArray:
    """Pack coordinates and Psi matrix into single flat array"""

    log.trace(f"Packing ray and beam parameters")

    beam_parameters = np.array(
        (q_X,
         q_Y,
         q_Z,
         K_X,
         K_Y,
         K_Z,
         np.real(Psi[0,0]), # Psi_xx, index 6
         np.real(Psi[1,1]), # Psi_yy, index 7
         np.real(Psi[2,2]), # Psi_zz, index 8
         np.real(Psi[0,1]), # Psi_xy, index 9
         np.real(Psi[0,2]), # Psi_xz, index 10
         np.real(Psi[1,2]), # Psi_yz, index 11
         np.imag(Psi[0,0]), # Psi_xx, index 12
         np.imag(Psi[1,1]), # Psi_yy, index 13
         np.imag(Psi[2,2]), # Psi_zz, index 14
         np.imag(Psi[0,1]), # Psi_xy, index 15
         np.imag(Psi[0,2]), # Psi_xz, index 16
         np.imag(Psi[1,2]), # Psi_yz, index 17
         )
    )

    return beam_parameters



def unpack_beam_parameters_3D(
        beam_parameters: FloatArray,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, FloatArray]:
    """Unpack the flat solver state vector into separate coordinate variables and Psi matrix"""

    log.trace(f"Unpacking ray and beam parameters")

    q_X = beam_parameters[0, ...]
    q_Y = beam_parameters[1, ...]
    q_Z = beam_parameters[2, ...]
    K_X = beam_parameters[3, ...]
    K_Y = beam_parameters[4, ...]
    K_Z = beam_parameters[5, ...]

    n_points = beam_parameters.shape[1] if beam_parameters.ndim == 2 else 1

    Psi = np.zeros([n_points, 3, 3], dtype="complex128")
    Psi[:,0,0] = beam_parameters[6,  ...] + 1j * beam_parameters[12, ...] # Psi_xx
    Psi[:,1,1] = beam_parameters[7,  ...] + 1j * beam_parameters[13, ...] # Psi_yy
    Psi[:,2,2] = beam_parameters[8,  ...] + 1j * beam_parameters[14, ...] # Psi_zz
    Psi[:,0,1] = beam_parameters[9,  ...] + 1j * beam_parameters[15, ...] # Psi_xy
    Psi[:,0,2] = beam_parameters[10, ...] + 1j * beam_parameters[16, ...] # Psi_xz
    Psi[:,1,2] = beam_parameters[11, ...] + 1j * beam_parameters[17, ...] # Psi_yz
    Psi[:,1,0] = Psi[:,0,1] # Psi is symmetric
    Psi[:,2,0] = Psi[:,0,2] # Psi is symmetric
    Psi[:,2,1] = Psi[:,1,2] # Psi is symmetric

    return q_X, q_Y, q_Z, K_X, K_Y, K_Z, np.squeeze(Psi)



def beam_evolution_fun_3D(tau, beam_parameters, hamiltonian: Hamiltonian_3D):
    """
    Parameters
    ----------
    tau : float
        Parameter along the ray.
    beam_parameters : complex128
        q_X, q_Y, q_Z, K_X, K_Y, K_Z, Psi_xx, Psi_yy, Psi_zz, Psi_xy, Psi_xz, Psi_yz

    Returns
    -------
    d_beam_parameters_d_tau
        d (beam_parameters) / d tau
    """

    q_X, q_Y, q_Z, K_X, K_Y, K_Z, Psi = unpack_beam_parameters_3D(beam_parameters)

    # Find derivatives of H
    dH = hamiltonian.derivatives(q_X, q_Y, q_Z, K_X, K_Y, K_Z, second_order = True)

    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians_3D(dH)
    grad_gradK_H = np.transpose(gradK_grad_H)

    dH_dX = dH["dH_dX"]
    dH_dY = dH["dH_dY"]
    dH_dZ = dH["dH_dZ"]
    dH_dKx = dH["dH_dKx"]
    dH_dKy = dH["dH_dKy"]
    dH_dKz = dH["dH_dKz"]

    d_Psi_d_tau = (- np.matmul(np.matmul(Psi, gradK_gradK_H), Psi)
                   - np.matmul(Psi, gradK_grad_H)
                   - np.matmul(grad_gradK_H, Psi)
                   - grad_grad_H)

    return pack_beam_parameters_3D(dH_dKx, dH_dKy, dH_dKz, -dH_dX, -dH_dY, -dH_dZ, d_Psi_d_tau)



def evolve_beam(q_initial_cartesian: FloatArray,
                K_initial_cartesian: FloatArray,
                Psi_3D_initial_labframe_cartesian: FloatArray,
                tau_leave: float,
                tau_points: FloatArray,
                hamiltonian: Hamiltonian_3D,
                rtol: float,
                atol: float):
    
    # Packing the initial q_X, q_Y, q_Z, K_X, K_Y, K_Z, and the components of Psi_3D
    beam_parameters_initial = pack_beam_parameters_3D(*q_initial_cartesian,
                                                      *K_initial_cartesian,
                                                      Psi_3D_initial_labframe_cartesian)

    # Evolving Psi_3D according to the beam-tracing equation
    log.info(f"Starting the beam solver")

    (solver_beam_output,
     duration_beam_solver) = timer(solve_ivp,
                                   beam_evolution_fun_3D,
                                   [0, tau_leave],
                                   beam_parameters_initial,
                                   method = "RK45",
                                   t_eval = tau_points,
                                   dense_output = False,
                                   events = None,
                                   vectorized = False,
                                   args = (hamiltonian,),
                                   rtol = rtol,
                                   atol = atol)
    
    log.info(f"""\n
        Beam solver status: {solver_beam_output.status}
        Beam solver took {duration_beam_solver} s
        Number of beam evolution evaluations: {solver_beam_output.nfev}
        Time per beam evolution evaluation: {duration_beam_solver / solver_beam_output.nfev}
    """)

    beam_solver_status = solver_beam_output.status
    tau_array = solver_beam_output.t
    beam_parameters_final = solver_beam_output.y

    (q_X_array, q_Y_array, q_Z_array,
     K_X_array, K_Y_array, K_Z_array,
     Psi_3D_output_labframe_cartesian) = unpack_beam_parameters_3D(beam_parameters_final)
    
    return (beam_solver_status, tau_array,
            q_X_array, q_Y_array, q_Z_array,
            K_X_array, K_Y_array, K_Z_array,
            Psi_3D_output_labframe_cartesian)