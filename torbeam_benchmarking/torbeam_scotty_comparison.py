import numpy as np
import matplotlib.pyplot as plt
import datatree
from torbeam_output import Torbeam_output
import scotty.plotting as plotting


def power_deposition_comparison(scotty_filepath: str, Torbeam_output: Torbeam_output):
    power_depo_scotty = np.loadtxt(scotty_filepath, dtype=float)
    rho = np.arange(0, 1, 1 / len(power_depo_scotty))
    plt.figure()
    plt.plot(Torbeam_output.rho, Torbeam_output.power_deposition, label = 'torbeam')
    plt.plot(rho, power_depo_scotty, label = 'scotty')
    plt.ylabel('dP/dV')
    plt.xlabel('rho')
    plt.title('Scotty Torbeam Power Deposition Comparison')
    plt.legend()


def current_drive_comparison(scotty_filepath: str, Torbeam_output: Torbeam_output):
    current_drive_scotty = np.loadtext(scotty_filepath, dtype=float)
    current_drive_scotty = np.zeros(100) # temporary, scotty output not available
    rho = np.arange(0, 1, 1 / len(current_drive_scotty))
    plt.figure()
    plt.plot(Torbeam_output.rho, Torbeam_output.current_drive, label = 'torbeam')
    plt.plot(rho, current_drive_scotty, label = 'scotty')
    plt.ylabel('J m')
    plt.xlabel('rho')
    plt.title('Scotty Torbeam Current Drive Comparison')
    plt.legend()


def beam_path_comparison(dt: datatree, Torbeam_output: Torbeam_output, x_range = [1, 3], y_range = [0, 1]):
    ax = plotting.plot_poloidal_beam_path(dt)
    ax.plot(Torbeam_output.q_R, Torbeam_output.q_Z, label = 'torbeam central ray', linestyle='--', dashes=(0.1, 0.1))
    ax.plot(Torbeam_output.upper_q_R, Torbeam_output.upper_q_Z, color = 'red', linestyle='--', dashes=(0.2, 0.1), label = "torbeam upper peripheral ray")
    ax.plot(Torbeam_output.lower_q_R, Torbeam_output.lower_q_Z, color = 'green', linestyle='--', dashes=(0.2, 0.1), label = 'torbeam lower peripheral ray')
    ax.legend(fontsize = "7")
    plt.xlim(x_range)
    plt.ylim(y_range)
    # plt.savefig('beam path comparison', dpi=1200)


def beam_length(x, y):
    z = [0.0]
    for i in range(1, len(x)):
        z.append(np.sqrt((x[i] - x[i-1]) ** 2 + (y[i] - y[i-1]) ** 2))
    for i in range(1, len(z)):
        z[i] += z[i-1]
    return np.array(z)


def Te_scaling(filename, scaling_factor: float):
    Te = np.fromfile(filename, dtype=float, sep="   ")
    
    format_string = "{:.12f}"
    for i in range(2, len(Te), 2):
        Te[i] *= scaling_factor
    f = open('Te.dat', 'w')
    f.writelines(['    ', str(int(Te[0])), '\n'])
    for i in range(1, len(Te), 2):
        f.writelines([format_string.format(Te[i]), '  ', str(Te[i + 1]), '\n'])
    f.close()