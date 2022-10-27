from scotty.fun_general import freq_GHz_to_wavenumber

from scipy.constants import c as speed_of_light
import numpy as np


def test_freq_GHz_to_wavenumber():
    assert np.isclose(freq_GHz_to_wavenumber(1.0 / (2 * np.pi)), 1e9 / speed_of_light)
