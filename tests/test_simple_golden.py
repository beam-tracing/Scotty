from scotty.beam_me_up import (
    beam_me_up,
    create_magnetic_geometry,
    make_density_fit,
)
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.torbeam import Torbeam
from scotty.geometry import CircularCrossSectionField
from scotty.fun_general import (
    freq_GHz_to_angular_frequency,
    angular_frequency_to_wavenumber,
)
from scotty.launch import find_entry_point, launch_beam

import json
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pathlib
import pytest

# Print more of arrays in failed tests
np.set_printoptions(linewidth=120, threshold=100)

CUTOFF_INDEX = 5

PSI_START_EXPECTED = np.array(
    [
        [
            -2.47357794e03 + 1.19557806e01j,
            -1.39916828e-02 - 1.43347874e-02j,
            3.40523926e02 - 1.13752568e02j,
        ],
        [
            -1.39916828e-02 - 1.43347874e-02j,
            4.24589207e03 + 4.35001002e03j,
            2.21127206e-03 + 2.26549697e-03j,
        ],
        [
            3.40523926e02 - 1.13752568e02j,
            2.21127206e-03 + 2.26549697e-03j,
            4.25776357e02 + 1.08229209e03j,
        ],
    ]
)
PSI_CUTOFF_EXPECTED = np.array(
    [
        [
            -3415.07623355 + 445.31992622j,
            -282.82924665 - 149.90119128j,
            509.18231115 - 398.50960651j,
        ],
        [
            -282.82924665 - 149.90119128j,
            3929.38936698 + 2642.61848492j,
            -55.70975677 - 74.68041153j,
        ],
        [
            509.18231115 - 398.50960651j,
            -55.70975677 - 74.68041153j,
            2049.26382135 + 377.83003374j,
        ],
    ]
)
PSI_FINAL_EXPECTED = np.array(
    [
        [
            -759.55423192 + 13243.111223j,
            -479.31204769 + 514.09773505j,
            -644.55622419 - 1452.45488266j,
        ],
        [
            -479.31204769 + 514.09773505j,
            3487.94478663 + 1885.63739326j,
            12.71272627 - 77.55460115j,
        ],
        [
            -644.55622419 - 1452.45488266j,
            12.71272627 - 77.55460115j,
            2470.84658744 + 159.68128543j,
        ],
    ]
)


EXPECTED = {
    "q_R_array": np.array(
        [
            1.99382564,
            1.8973136,
            1.82080896,
            1.76044632,
            1.71327164,
            1.69395316,
            1.67708778,
            1.65019221,
            1.63108066,
            1.61859347,
        ]
    ),
    "q_Z_array": np.array(
        [
            -0.07804514,
            -0.09137557,
            -0.10919534,
            -0.13253917,
            -0.1627768,
            -0.18095369,
            -0.20155828,
            -0.25072523,
            -0.3125676,
            -0.3901351,
        ]
    ),
    "q_zeta_array": np.array(
        [
            0.0,
            -0.00041565,
            -0.00168641,
            -0.00380195,
            -0.00660756,
            -0.00815794,
            -0.00972988,
            -0.01265907,
            -0.01496556,
            -0.01641859,
        ]
    ),
    "K_R_array": np.array(
        [
            -1146.40007,
            -909.7351259,
            -718.97020274,
            -564.32641187,
            -437.85637809,
            -383.2161993,
            -333.12907589,
            -245.07498321,
            -169.39921007,
            -102.43869197,
        ]
    ),
    "K_Z_array": np.array(
        [
            -120.49150267,
            -165.72589293,
            -219.64365799,
            -284.80168007,
            -364.33147671,
            -410.55892646,
            -462.07152722,
            -582.78670158,
            -732.58798218,
            -919.20759736,
        ]
    ),
    "K_zeta_initial": np.array(-0.0),
    "tau_array": np.array(
        [
            0.0,
            62.6209717,
            125.2419434,
            187.86291511,
            250.48388681,
            281.73236206,
            313.10485851,
            375.72583021,
            438.34680191,
            500.96777362,
        ]
    ),
    "B_magnitude": np.array(
        [
            0.75898041,
            0.79758802,
            0.83110015,
            0.85959713,
            0.883266,
            0.8933391,
            0.90232283,
            0.91702929,
            0.92777423,
            0.93493185,
        ]
    ),
    "normalised_gyro_freqs": np.array(
        [
            0.38628639,
            0.4059359,
            0.42299204,
            0.43749571,
            0.44954208,
            0.45466883,
            0.45924114,
            0.46672606,
            0.47219474,
            0.47583764,
        ]
    ),
    "normalised_plasma_freqs": np.array(
        [
            0.0,
            0.59773859,
            0.75915389,
            0.83777097,
            0.8712485,
            0.87520137,
            0.8712369,
            0.83771518,
            0.7590454,
            0.59751663,
        ]
    ),
}


def simple(path):
    """Built-in synthetic diagnostic"""
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["find_B_method"] = "unit-tests"
    return kwargs_dict


def ne_dat_file(path):
    """Density fit using TORBEAM file"""
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")

    ne_filename = path / "ne.dat"
    rho = np.linspace(0, 1)
    density_fit = kwargs_dict["density_fit_method"]
    density = density_fit(rho**2)

    with open(ne_filename, "w") as f:
        f.write(f"{len(rho)}\n")
        np.savetxt(f, np.column_stack((rho, density)), fmt="%.7e")

    kwargs_dict["find_B_method"] = "unit-tests"
    kwargs_dict["density_fit_parameters"] = None
    kwargs_dict["density_fit_method"] = "smoothing-spline-file"
    kwargs_dict["ne_data_path"] = path
    return kwargs_dict


def synthetic_args():
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    field = CircularCrossSectionField(
        R_axis=kwargs_dict["R_axis"],
        minor_radius_a=kwargs_dict["minor_radius_a"],
        B_T_axis=kwargs_dict["B_T_axis"],
        B_p_a=kwargs_dict["B_p_a"],
        R_points=256,
        Z_points=256,
        grid_buffer_factor=1.1,
    )
    x_meshgrid, z_meshgrid = np.meshgrid(field.R_coord, field.Z_coord, indexing="ij")
    B_t = field.B_T(x_meshgrid, z_meshgrid)
    B_r = field.B_R(x_meshgrid, z_meshgrid)
    B_z = field.B_Z(x_meshgrid, z_meshgrid)
    psi = field.poloidal_flux(x_meshgrid, z_meshgrid)
    return kwargs_dict, B_r, B_t, B_z, psi, field


def torbeam_file(path):
    """Geometry using TORBEAM file"""
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()
    x_meshgrid, z_meshgrid = np.meshgrid(field.R_coord, field.Z_coord, indexing="ij")
    B_t = field.B_T(x_meshgrid, z_meshgrid)
    B_r = field.B_R(x_meshgrid, z_meshgrid)
    B_z = field.B_Z(x_meshgrid, z_meshgrid)
    psi = field.poloidal_flux(x_meshgrid, z_meshgrid)

    Torbeam(field.R_coord, field.Z_coord, B_r, B_t, B_z, psi).write(path / "topfile")

    kwargs_dict["find_B_method"] = "torbeam"
    kwargs_dict["magnetic_data_path"] = path
    return kwargs_dict


def omfit_json(path):
    """Geometry using OMFIT JSON file"""
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    def flatten_as_F(array):
        return array.T.flatten().tolist()

    data = {
        "R": field.R_coord.tolist(),
        "Z": field.Z_coord.tolist(),
        "Br": flatten_as_F(B_R),
        "Bt": flatten_as_F(B_t),
        "Bz": flatten_as_F(B_Z),
        "pol_flux": flatten_as_F(psi),
    }

    with open(path / "topfile.json", "w") as f:
        json.dump(data, f)

    kwargs_dict["find_B_method"] = "omfit"
    kwargs_dict["magnetic_data_path"] = path
    return kwargs_dict


def add_timeslices(array):
    """Add a new dimension at the front, with 'array' at index 1"""
    zeros = np.zeros_like(array)
    return np.stack((zeros, array, zeros))


def npz_file(path: pathlib.Path):
    """find_B_method = "test" """

    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    time = [100, 200, 300]

    shot = 12345
    np.savez(
        path / f"{shot}_equilibrium_data",
        R_EFIT=field.R_coord,
        Z_EFIT=field.Z_coord,
        poloidalFlux_grid=add_timeslices(psi),
        Bphi_grid=add_timeslices(B_t),
        Br_grid=add_timeslices(B_R),
        Bz_grid=add_timeslices(B_Z),
        time_EFIT=time,
    )
    kwargs_dict["find_B_method"] = "test"
    kwargs_dict["magnetic_data_path"] = path
    kwargs_dict["shot"] = shot
    kwargs_dict["equil_time"] = time[1]

    return kwargs_dict


def npz_notime_file(path: pathlib.Path):
    """find_B_method = "test_notime" """
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    np.savez(
        path / "12345_equilibrium_data",
        R_EFIT=field.R_coord,
        Z_EFIT=field.Z_coord,
        poloidalFlux_grid=psi,
        Bphi_grid=B_t,
        Br_grid=B_R,
        Bz_grid=B_Z,
    )
    kwargs_dict["find_B_method"] = "test_notime"
    kwargs_dict["magnetic_data_path"] = path / "12345_equilibrium_data.npz"
    return kwargs_dict


def UDA_saved(path: pathlib.Path):
    """find_B_method = "UDA_saved" with shot = None"""
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]
    B_p_a = kwargs_dict["B_p_a"]
    psi_axis = field.poloidal_flux(R_axis, 0.0) * B_p_a
    psi_boundary = field.poloidal_flux(R_axis + minor_radius, 0.0) * B_p_a

    R_midplane = np.linspace(R_axis, R_axis + minor_radius)
    rBphi = R_midplane * field.B_T(R_midplane, 0.0)

    np.savez(
        path / "12345_equilibrium_data",
        rBphi=rBphi,
        R_EFIT=field.R_coord,
        Z_EFIT=field.Z_coord,
        poloidalFlux_grid=psi,
        poloidalFlux_unnormalised_axis=-psi_axis,
        poloidalFlux_unnormalised_boundary=-psi_boundary,
    )
    kwargs_dict["find_B_method"] = "UDA_saved"
    kwargs_dict["magnetic_data_path"] = path / "12345_equilibrium_data.npz"
    kwargs_dict["shot"] = None
    kwargs_dict["delta_R"] = -0.0001
    kwargs_dict["delta_Z"] = 0.0001
    return kwargs_dict


def UDA_saved_MAST_U(path: pathlib.Path):
    """find_B_method = "UDA_saved" with shot = None"""
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    shot = 55555

    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]
    B_p_a = kwargs_dict["B_p_a"]
    psi_axis = field.poloidal_flux(R_axis, 0.0) * B_p_a
    psi_boundary = field.poloidal_flux(R_axis + minor_radius, 0.0) * B_p_a

    R_midplane = np.linspace(R_axis, R_axis + minor_radius)
    rBphi = R_midplane * field.B_T(R_midplane, 0.0)
    psi_norm_1D = np.linspace(0, 1, len(R_midplane))

    time = [100, 200, 300]

    np.savez(
        path / f"{shot}_equilibrium_data",
        rBphi=add_timeslices(rBphi),
        R_EFIT=field.R_coord,
        Z_EFIT=field.Z_coord,
        poloidalFlux_grid=add_timeslices(psi),
        poloidalFlux=add_timeslices(psi_norm_1D),
        poloidalFlux_unnormalised_axis=add_timeslices(-psi_axis),
        poloidalFlux_unnormalised_boundary=add_timeslices(-psi_boundary),
        time_EFIT=time,
    )
    kwargs_dict["find_B_method"] = "UDA_saved"
    kwargs_dict["magnetic_data_path"] = path
    kwargs_dict["shot"] = shot
    kwargs_dict["delta_R"] = -0.0001
    kwargs_dict["delta_Z"] = 0.0001
    kwargs_dict["equil_time"] = time[1]
    return kwargs_dict


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
        # Following methods have errors of 30%, despite field being
        # idential to 1%, needs investigating
        # pytest.param(UDA_saved, id="UDA-saved-file"),
    ],
)
def test_integrated(tmp_path, generator):
    """Golden answer test to check basic functionality using circular
    flux surfaces."""

    kwargs_dict = generator(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa0.10"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path

    number_of_existing_npz_files = len(list(tmp_path.glob("*.npz")))

    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4 + number_of_existing_npz_files

    with np.load(tmp_path / "data_output_Bpa0.10.npz") as f:
        output = dict(f)

    for key, value in EXPECTED.items():
        assert_allclose(output[key], value, rtol=1e-2, atol=1e-2, err_msg=key)

    K_magnitude = np.hypot(output["K_R_array"], output["K_Z_array"])
    assert K_magnitude.argmin() == CUTOFF_INDEX

    assert_allclose(
        output["Psi_3D_output"][0, ...], PSI_START_EXPECTED, rtol=1e-2, atol=0.1
    )
    assert_allclose(
        output["Psi_3D_output"][CUTOFF_INDEX, ...],
        PSI_CUTOFF_EXPECTED,
        rtol=1e-2,
        atol=0.1,
    )
    assert_allclose(
        output["Psi_3D_output"][-1, ...], PSI_FINAL_EXPECTED, rtol=1.8e-2, atol=0.1
    )


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(omfit_json, id="omfit-file"),
        pytest.param(npz_file, id="test-file"),
        pytest.param(npz_notime_file, id="test_notime-file"),
        pytest.param(UDA_saved, id="UDA-saved-file"),
        pytest.param(UDA_saved_MAST_U, id="UDA-saved-MAST-U-file"),
    ],
)
def test_create_magnetic_geometry(tmp_path, generator):
    kwargs_dict = generator(tmp_path)

    field = create_magnetic_geometry(**kwargs_dict)
    field_golden = create_magnetic_geometry(**simple(tmp_path))

    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]
    R = np.linspace(R_axis + 0.1, R_axis + minor_radius, 10)
    Z = np.linspace(0.1, kwargs_dict["minor_radius_a"], 10)

    assert_allclose(
        field.B_R(R_axis, Z), field_golden.B_R(R_axis, Z), rtol=1e-3, atol=1e-3
    )
    assert_allclose(field.B_T(R, 0.0), field_golden.B_T(R, 0.0), rtol=1e-3, atol=1e-3)
    assert_allclose(field.B_Z(R, 0.0), field_golden.B_Z(R, 0.0), rtol=1e-3, atol=1e-3)
    assert_allclose(
        field.poloidal_flux(R, 0.0),
        field_golden.poloidal_flux(R, 0.0),
        rtol=1e-3,
        atol=1e-3,
    )


def test_EFIT_geometry(tmp_path):
    kwargs_dict = UDA_saved(tmp_path)
    field = create_magnetic_geometry(**kwargs_dict)
    field_golden = create_magnetic_geometry(**simple(tmp_path))

    R_axis = kwargs_dict["R_axis"]
    minor_radius_a = kwargs_dict["minor_radius_a"]
    B_p_a = kwargs_dict["B_p_a"]

    # Include buffer for gradient near edge
    width = minor_radius_a + 0.05
    # Different grid sizes to capture transpose errors
    R = np.linspace(R_axis - width, R_axis + width, 99)
    Z = np.linspace(-width, width, 101)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")
    B_R = field.B_R(R_grid, Z_grid)
    B_Z = field.B_Z(R_grid, Z_grid)

    psi = field.poloidal_flux(R_grid, Z_grid)
    grad_psi = np.gradient(psi, R, Z)
    calculated_B_R = grad_psi[1] * B_p_a / R_grid
    calculated_B_Z = -grad_psi[0] * B_p_a / R_grid

    mask = (psi > 0.1) & (psi <= 1)

    assert_allclose(B_R[mask], calculated_B_R[mask], 2e-3, 2e-3)
    assert_allclose(B_Z[mask], calculated_B_Z[mask], 2e-3, 2e-3)

    assert_allclose(field.B_T(R_grid, Z_grid), field_golden.B_T(R_grid, Z_grid))

    # Check that poloidal field rotates in the correct direction
    total_sign = np.sign(B_R) * np.sign(B_Z)
    assert total_sign[0, 0] == -1, "Top left"
    assert total_sign[0, -1] == 1, "Top right"
    assert total_sign[-1, 0] == 1, "Bottom left"
    assert total_sign[-1, -1] == -1, "Bottom right"


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(omfit_json, id="omfit-file"),
        pytest.param(npz_file, id="test-file"),
        pytest.param(npz_notime_file, id="test_notime-file"),
        # pytest.param(UDA_saved, id="UDA-saved-file"),
        # pytest.param(UDA_saved_MAST_U, id="UDA-saved-MAST-U-file"),
    ],
)
def test_launch_golden_answer(tmp_path, generator):
    args = generator(tmp_path)
    field = create_magnetic_geometry(**args)
    ne_data_path = args.get("ne_data_path", None)
    if ne_data_path:
        ne_filename = ne_data_path / "ne.dat"
        parameters = [ne_filename, 5, 0]
    else:
        parameters = None
        ne_filename = None

    density_fit = make_density_fit(
        args["density_fit_method"],
        args["poloidal_flux_enter"],
        parameters,
        ne_filename,
    )

    launch_angular_frequency = freq_GHz_to_angular_frequency(args["launch_freq_GHz"])

    (
        K_initial,
        initial_position,
        launch_K,
        Psi_3D_lab_initial,
        Psi_3D_lab_launch,
        Psi_3D_lab_entry,
        Psi_3D_lab_entry_cartersian,
        distance_from_launch_to_entry,
    ) = launch_beam(
        field=field,
        find_density_1D=density_fit,
        toroidal_launch_angle_Torbeam=args["toroidal_launch_angle_Torbeam"],
        poloidal_launch_angle_Torbeam=args["poloidal_launch_angle_Torbeam"],
        mode_flag=args["mode_flag"],
        launch_beam_width=args["launch_beam_width"],
        launch_beam_curvature=args["launch_beam_curvature"],
        launch_position=args["launch_position"],
        vacuum_propagation_flag=args["vacuum_propagation_flag"],
        Psi_BC_flag=args["Psi_BC_flag"],
        poloidal_flux_enter=args["poloidal_flux_enter"],
        launch_angular_frequency=launch_angular_frequency,
    )

    expected_K_initial = np.array([-1146.4000699962507, -0.0, -120.4915026654739])
    expected_initial_position = np.array([1.99382564, -0.0, -0.07804514])
    expected_launch_K = np.array([-1146.40007, -0.0, -120.49150267])
    # Note that R-zeta, Z-zeta terms are not actually zero, but should be
    expected_Psi_3D_lab_initial = np.array(
        [
            [-2.473578e03 + 1.195578e01j, 0.0 + 0.0j, 3.405239e02 - 1.137526e02j],
            [0.0 + 0.0j, 4.245892e03 + 4.350010e03j, 0.0 + 0.0j],
            [3.405239e02 - 1.137526e02j, 0.0 + 0.0j, 4.257764e02 + 1.082292e03j],
        ]
    )
    expected_Psi_3D_lab_launch = np.array(
        [
            [-3.1486979 + 13.65774954j, 0.0 + 0.0j, 29.9578594 - 129.94480676j],
            [0.0 + 0.0j, 1037.08121046 + 8365.71125j, 0.0 + 0.0j],
            [29.9578594 - 129.94480676j, 0.0 + 0.0j, -285.02999262 + 1236.34225046j],
        ]
    )
    expected_Psi_3D_lab_entry = np.array(
        [
            [5.38751589 + 11.95597601j, 0.0 + 0.0j, -51.25878964 - 113.75351313j],
            [0.0 + 0.0j, 4245.89207422 + 4350.0100231j, 0.0 + 0.0j],
            [-51.25878964 - 113.75351313j, 0.0 + 0.0j, 487.69480611 + 1082.29238188j],
        ]
    )
    expected_Psi_3D_lab_entry_cartersian = np.array(
        [
            [5.38751589 + 11.95597601j, 0.0 + 0.0j, -51.25878964 - 113.75351313j],
            [0.0 + 0.0j, 493.082322 + 1094.24835789j, 0.0 + 0.0j],
            [-51.25878964 - 113.75351313j, 0.0 + 0.0j, 487.69480611 + 1082.29238188j],
        ]
    )
    expected_distance_from_launch_to_entry = 0.5964417281350541
    expected_Psi_3D = np.array(
        [
            [-3.1486979 + 13.65774954j, 0.0 + 0.0j, 29.9578594 - 129.94480676j],
            [0.0 + 0.0j, 1037.08121046 + 8365.71125j, 0.0 + 0.0j],
            [29.9578594 - 129.94480676j, 0.0 + 0.0j, -285.02999262 + 1236.34225046j],
        ]
    )

    tol = 1e-3
    assert_allclose(Psi_3D_lab_launch, expected_Psi_3D, tol, tol)
    assert_allclose(K_initial, expected_K_initial, tol, tol)
    assert_allclose(initial_position, expected_initial_position, tol, tol)
    assert_allclose(launch_K, expected_launch_K, tol, tol)
    assert_allclose(Psi_3D_lab_launch, expected_Psi_3D_lab_launch, tol, tol)
    assert_allclose(Psi_3D_lab_entry, expected_Psi_3D_lab_entry, tol, tol)
    assert_allclose(
        Psi_3D_lab_entry_cartersian, expected_Psi_3D_lab_entry_cartersian, tol, tol
    )
    assert_allclose(
        distance_from_launch_to_entry,
        expected_distance_from_launch_to_entry,
        tol,
        tol,
    )
    assert_allclose(launch_K, expected_launch_K, tol, tol)
    assert_allclose(Psi_3D_lab_launch, expected_Psi_3D_lab_launch, tol, tol)
    assert_allclose(Psi_3D_lab_entry, expected_Psi_3D_lab_entry, tol, tol)
    assert_allclose(
        Psi_3D_lab_entry_cartersian, expected_Psi_3D_lab_entry_cartersian, tol, tol
    )
    # Larger atol due to some small values
    assert_allclose(Psi_3D_lab_initial, expected_Psi_3D_lab_initial, tol, atol=0.1)


def launch_parameters(start_point, end_point_poloidal_coords):
    kwargs_dict = simple(None)
    rho_end, theta_end, zeta_end = end_point_poloidal_coords
    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]

    R_end = R_axis + rho_end * minor_radius * np.cos(theta_end)
    Z_end = rho_end * minor_radius * np.sin(theta_end)
    X_end = R_end * np.cos(zeta_end)
    Y_end = R_end * np.sin(zeta_end)

    X_start, Y_start, Z_start = start_point
    phi_t = np.arctan2(Y_end - Y_start, X_end - X_start) - np.pi
    poloidal_length = np.sqrt((Y_end - Y_start) ** 2 + (X_end - X_start) ** 2)
    phi_p = np.arctan2(Z_end - Z_start, poloidal_length)
    expected_entry_cartesian = (R_end, zeta_end, Z_end)
    return start_point, -phi_p, phi_t, expected_entry_cartesian


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(simple, id="simple"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
        pytest.param(UDA_saved, id="UDA-saved-file"),
    ],
)
@pytest.mark.parametrize(
    (
        "launch_position",
        "poloidal_launch_angle",
        "toroidal_launch_angle",
        "expected_entry",
    ),
    (
        pytest.param([2.5, 0, 0], 0, 0, [2.0, 0.0, 0], id="outboard-midplane"),
        pytest.param([0.5, 0, 0], 0, np.pi, [1.0, 0, 0], id="inboard-midplane"),
        pytest.param([1.5, 0, 1], np.pi / 2, 0, [1.5, 0, 0.5], id="top"),
        pytest.param([1.5, 0, -1], -np.pi / 2, 0, [1.5, 0, -0.5], id="bottom"),
        pytest.param(
            [2, 0, 0.5],
            np.pi / 4,
            0,
            [1.5 + np.sqrt(0.5) / 2, 0, np.sqrt(0.5) / 2],
            id="top-right",
        ),
        pytest.param(
            [2, 0, -0.5],
            -np.pi / 4,
            0,
            [1.5 + np.sqrt(0.5) / 2, 0, -np.sqrt(0.5) / 2],
            id="bottom-right",
        ),
        pytest.param(
            *launch_parameters((2.5, 0.0, -0.1), (1, np.pi / 4, np.pi / 8)),
            id="steep-angle",
        ),
    ),
)
def test_find_entry_point(
    tmp_path,
    generator,
    launch_position,
    poloidal_launch_angle,
    toroidal_launch_angle,
    expected_entry,
):
    args = generator(tmp_path)
    field = create_magnetic_geometry(**args)
    poloidal_flux_enter = args["poloidal_flux_enter"]

    entry_position = find_entry_point(
        launch_position,
        poloidal_launch_angle,
        toroidal_launch_angle,
        poloidal_flux_enter,
        field,
    )

    assert_allclose(entry_position, expected_entry, 1e-6, 1e-6)
