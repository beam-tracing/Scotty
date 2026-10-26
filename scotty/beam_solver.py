import logging
import numpy as np
from scipy.integrate import solve_ivp
from scotty.checks import VALID_SOLVER_STATUS
from scotty.geometry import MagneticField_Cartesian
from scotty.hamiltonian import Hamiltonian, hessians
from scotty.logger import timer
from scotty.typing import FloatArray, ComplexFloatArray
from typing import Tuple

log = logging.getLogger(__name__)

I, J = np.triu_indices(3)


def pack_beam_parameters(
    q: FloatArray, K: FloatArray, Psi: ComplexFloatArray
) -> FloatArray:
    """Packs `q` (shape `(3, N)`), `K` (shape `(3, N)`), and `Psi`
    (shape `(N, 3, 3)`) into an array of shape `(18, N)`"""

    log.trace(f"Packing ray and beam parameters")

    # Validity checks
    qshape = q.shape
    Kshape = K.shape
    Psishape = Psi.shape
    if not q.shape == K.shape:
        raise ValueError(
            f"`q` and `K` must have the same shape but got shapes {qshape} and {Kshape}"
        )
    if not q.ndim == Psi.ndim - 1:
        raise ValueError(
            f"Each point of `q` must have a corresponding `Psi`, but got shapes {qshape} and {Psishape}"
        )
    if q.ndim not in [1, 2] or qshape[0] != 3:
        raise ValueError(f"`q` must have shape `(3,)` or `(3,N)` but got {qshape}")
    if K.ndim not in [1, 2] or Kshape[0] != 3:
        raise ValueError(f"`K` must have shape `(3,)` or `(3,N)` but got {Kshape}")

    if Psi.ndim == 3:
        upper_Psi = Psi[:, I, J].T
    else:
        upper_Psi = Psi[I, J]

    shape = 18 if q.ndim == 1 else (18, q.shape[1])
    q_K_RePsi_ImPsi_arr = np.empty(shape, dtype=np.float64)
    q_K_RePsi_ImPsi_arr[0:3] = q
    q_K_RePsi_ImPsi_arr[3:6] = K
    q_K_RePsi_ImPsi_arr[6:12] = (
        upper_Psi.real
    )  # Psi[0,0].real, Psi[1,0].real, Psi[2,0].real, Psi[1,1].real, Psi[1,2].real, Psi[2,2].real
    q_K_RePsi_ImPsi_arr[12:18] = (
        upper_Psi.imag
    )  # Psi[0,0].imag, Psi[1,0].imag, Psi[2,0].imag, Psi[1,1].imag, Psi[1,2].imag, Psi[2,2].imag

    return q_K_RePsi_ImPsi_arr


def unpack_beam_parameters(
    q_K_RePsi_ImPsi_arr: FloatArray, transpose_flag: bool = False
) -> Tuple[FloatArray, FloatArray, ComplexFloatArray]:
    """Unpacks `q_K_RePsi_ImPsi_arr` of shape `(18, N)` back into
    `q` (shape `(3, N)`), `K` (shape `(3, N)`), and `Psi` (shape `(N, 3, 3)`)"""

    log.trace(f"Unpacking ray and beam parameters")
    q = q_K_RePsi_ImPsi_arr[0:3]
    K = q_K_RePsi_ImPsi_arr[3:6]
    Psi_upper = q_K_RePsi_ImPsi_arr[6:12] + 1j * q_K_RePsi_ImPsi_arr[12:18]

    shape = (
        (3, 3)
        if q_K_RePsi_ImPsi_arr.ndim == 1
        else (q_K_RePsi_ImPsi_arr.shape[1], 3, 3)
    )
    Psi = np.empty(shape, dtype=np.complex128)
    if q_K_RePsi_ImPsi_arr.ndim == 1:
        Psi[I, J] = Psi_upper
        Psi[J, I] = Psi_upper
    else:
        Psi[:, I, J] = Psi_upper.T
        Psi[:, J, I] = Psi[:, I, J]

    if transpose_flag:
        return q.T, K.T, Psi  # (N,3), (N,3), (N,3,3)
    else:
        return q, K, Psi  # (3,N), (3,N), (N,3,3)


def d_beam_parameters_d_tau(
    tau: FloatArray, beam_parameters: FloatArray, hamiltonian: Hamiltonian
) -> FloatArray:
    """
    something
    """

    # tau is unused here, but solve_ivp is strict about the function signatures and requires something like
    # f(t, y, y0), so we put the tau here so that it stops complaining

    q, K, Psi = unpack_beam_parameters(beam_parameters)

    dH = hamiltonian.derivatives(q, K, second_order=True)
    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(
        dH=dH, cartesian=isinstance(hamiltonian.field, MagneticField_Cartesian)
    )

    d = np.array(list(dH.values())[:6])
    dq_dtau, dK_dtau = d[3:], -d[:3]

    dPsi_dtau = (
        -np.matmul(np.matmul(Psi, gradK_gradK_H), Psi)
        - (Psi_gradK_grad_H := np.matmul(Psi, gradK_grad_H))
        - np.transpose(Psi_gradK_grad_H)
        - grad_grad_H
    )

    return pack_beam_parameters(dq_dtau, dK_dtau, dPsi_dtau)


def beam_tracing(
    tau_leave: float,
    tau_points: FloatArray,
    q_initial: FloatArray,
    K_initial: FloatArray,
    Psi_3D_initial_labframe: ComplexFloatArray,
    hamiltonian: Hamiltonian,
    rtol: float,
    atol: float,
) -> Tuple[
    VALID_SOLVER_STATUS,
    int,
    float,
    FloatArray,
    FloatArray,
    FloatArray,
    ComplexFloatArray,
]:
    """Returns solver status (1, 0, -1), duration taken by solver, number of evaluations,
    tau` of shape `(N,)`, `q` of shape `(3, N)`, `K` of shape `(3, N)`, `Psi` of shape `(N, 3, 3)`
    """

    # Packing the ray and beam parameters
    beam_parameters_initial = pack_beam_parameters(
        q_initial, K_initial, Psi_3D_initial_labframe
    )

    # Evolving Psi_3D by solving the beam-tracing equation
    log.info(f"Starting the beam solver")

    solver_beam_output, duration_beam_solver = timer(solve_ivp)(
        fun=d_beam_parameters_d_tau,
        t_span=[0, tau_leave],
        y0=beam_parameters_initial,
        method="RK45",
        t_eval=tau_points,
        dense_output=False,
        events=None,
        vectorized=False,
        args=(hamiltonian,),
        rtol=rtol,
        atol=atol,
    )

    log.info(f"""\n
        Beam solver status: {solver_beam_output.status}
        Beam solver took {duration_beam_solver} s
        Number of beam evolution evaluations: {solver_beam_output.nfev}
        Time per beam evolution evaluation: {duration_beam_solver / solver_beam_output.nfev}
    """)

    return (
        solver_beam_output.status,
        solver_beam_output.nfev,
        duration_beam_solver,
        solver_beam_output.t,
        *unpack_beam_parameters(solver_beam_output.y, transpose_flag=True),
    )
