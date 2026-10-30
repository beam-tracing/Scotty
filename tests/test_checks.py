import pytest

from scotty.checks import Parameters


def _check_flux(enter, zero_density, zero_temperature):
    # The checks only use the class's helper methods, so no full Parameters set-up is needed
    return Parameters._check_poloidal_flux_arguments(
        Parameters.__new__(Parameters), enter, zero_density, zero_temperature, False
    )


def test_poloidal_flux_equal_values_accepted():
    assert _check_flux(1.3, 1.3, 1.3) == (1.3, 1.3, 1.3)


def test_poloidal_flux_zero_density_below_enter_names_density():
    with pytest.raises(ValueError) as err:
        _check_flux(1.3, 1.0, 1.3)
    message = str(err.value)
    assert "`poloidal_flux_zero_density` must be greater than or equal to" in message
    assert "= 1.0" in message


def test_poloidal_flux_zero_temperature_below_enter_names_temperature():
    with pytest.raises(ValueError) as err:
        _check_flux(1.3, 1.3, 1.0)
    message = str(err.value)
    assert "`poloidal_flux_zero_temperature` must be greater than or equal to" in message
    assert "`poloidal_flux_zero_density`" not in message
    assert "= 1.0" in message


def _circular_flag(**extras):
    params = Parameters.__new__(Parameters)
    params._unpack_kwargs_for_circular_flux_surfaces(extras=extras)
    return params.circular_flux_surfaces_flag


def test_circular_flux_surfaces_accept_zero_poloidal_field():
    # B_p_a = 0 (no poloidal field) is a valid value, not a missing one
    assert _circular_flag(B_T_axis=1.0, B_p_a=0.0, R_axis=1.5, minor_radius_a=0.5)


def test_circular_flux_surfaces_need_all_or_none():
    assert not _circular_flag()
    with pytest.raises(ValueError):
        _circular_flag(B_T_axis=1.0, R_axis=1.5)
