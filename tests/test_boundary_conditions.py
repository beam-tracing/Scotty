# import numpy as np
# from scotty.boundary_conditions import find_Psi_3D_plasma

# # TODO
# def test_find_Psi_3D_plasma_discontinuous():
#     Psi_vacuum_3D = np.array(
#         [
#             [
#                 22.96480274 + 25.35603703j,
#                 -288.34856095 - 200.18496508j,
#                 -76.01536319 - 83.93054303j,
#             ],
#             [
#                 -288.34856095 - 200.18496508j,
#                 3901.7586168 + 2724.12830862j,
#                 -22.50617362 - 24.84965268j,
#             ],
#             [
#                 -76.01536319 - 83.93054303j,
#                 -22.50617362 - 24.84965268j,
#                 625.89678142 + 691.06894368j,
#             ],
#         ]
#     )
#     K_v_R = -720.1663330442489
#     K_v_zeta = -213.22253614625006
#     K_v_Z = -89.39673984646942
#     K_p_R = -572.3610161829057
#     K_p_zeta = -213.22253614625006
#     K_p_Z = -116.25073909098793
#     dH_dKR = -0.0021297263342362482
#     dH_dKzeta = -0.00018858310699032543
#     dH_dKZ = -0.00044100184796613817
#     dH_dR = -2.2863224601654686
#     dH_dZ = 0.40208647341111176
#     dpolflux_dR = 1.9677859515843108
#     dpolflux_dZ = -0.3575170608158262
#     d2polflux_dR2 = 0.12782996883231587
#     d2polflux_dZ2 = 3.8721770234673154
#     d2polflux_dRdZ = 0.7035172444602721

#     Psi_3D_plasma = find_Psi_3D_plasma_discontinuous(
#         Psi_vacuum_3D,
#         K_v_R,
#         K_v_zeta,
#         K_v_Z,
#         K_p_R,
#         K_p_zeta,
#         K_p_Z,
#         dH_dKR,  # In the plasma
#         dH_dKzeta,  # In the plasma
#         dH_dKZ,  # In the plasma
#         dH_dR,  # In the plasma
#         dH_dZ,  # In the plasma
#         dpolflux_dR,  # Continuous
#         dpolflux_dZ,  # Continuous
#         d2polflux_dR2,  # Continuous
#         d2polflux_dZ2,  # Continuous
#         d2polflux_dRdZ,  # Continuous
#     )

#     expected_Psi_3D_plasma = np.array(
#         [
#             [
#                 -1035.11039468 + 51.2582456j,
#                 -458.03354529 - 237.47347794j,
#                 10.33282589 - 145.99156328j,
#             ],
#             [
#                 -458.03354529 - 237.47347794j,
#                 3901.7586168 + 2724.12830862j,
#                 543.49055857 - 18.07489168j,
#             ],
#             [
#                 10.33282589 - 145.99156328j,
#                 543.49055857 - 18.07489168j,
#                 629.44689491 + 712.76503166j,
#             ],
#         ]
#     )

#     npt.assert_allclose(Psi_3D_plasma, expected_Psi_3D_plasma)