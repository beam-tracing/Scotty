import numpy as np

from scotty.fun_general import read_floats_into_list_until


"""
Functions in this file:
1. psi_rescaling: rescale the poloidal flux to match Scotty format
2. topfile_reformatting: reformat the topfile to Scotty format
3. beam_length: calculate the length of the beam (for torbeam)
4. torbeam_in_plasma: find the entry and exit point of torbeam in the plasma
5. Te_scaling: rescale the electron temperature profile
"""


def psi_rescaling(input_path, output_path):
    """
    
    Topfile directly from Torbeam has a different scaling for poloidal flux, need to
    rescale it to scotty format (0 at centre and 1 at the last closed flux surface)
    
    
    """
    input_file = input_path + "\\topfile"
    output_file = output_path + "\\topfile"

    # Suggest that output_file in a different directory to avoid overwriting
    # calculating the scaling factor(find max and min psi)
    with open(input_file) as f:
        while "Inside and Outside radius and psi_sep" not in f.readline():
            pass  # Start reading only from this onwards
        psi_sep = read_floats_into_list_until("X-coordinates", f)[2]
        
        
        while "psi" not in f.readline():
            pass  #

        poloidal_flux = np.array(read_floats_into_list_until("you fall asleep", f))
    
    # some flux data has maximum at the centre, some has minimum, needs to distinguish
    if np.abs(poloidal_flux[-1] - poloidal_flux.min()) > np.abs(poloidal_flux[-1] - poloidal_flux.max()):
        poloidal_flux_centre = poloidal_flux.min()
    else:
        poloidal_flux_centre = poloidal_flux.max()
    
    
    # writing the change
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()

    new_lines = []
    in_scaling_section = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == 'psi':
            in_scaling_section = True
            new_lines.append(line)
            continue
        elif 'you fall asleep' in stripped_line:
            in_scaling_section = False
            new_lines.append(line)
            continue
        if in_scaling_section:
            numbers = stripped_line.split()
            scaled_numbers = []
            for num in numbers:
                try:
                    # rescale poloidal_flux to match scotty format
                    # poloidal_flux.max() -> 0; psi_sep -> 1
                    scaled_num = (float(num) - poloidal_flux_centre) / (
                        psi_sep - poloidal_flux_centre)
                    scaled_numbers.append(str(scaled_num))
                except ValueError:
                    # Keep non-numeric tokens unchanged
                    scaled_numbers.append(num)
            new_line = '  '.join(scaled_numbers) + '\n'
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(output_file, 'w') as f_out:
        f_out.writelines(new_lines)


def topfile_reformatting(input_path, output_path):
    # Reformat topfile to Scotty format
    input_file = input_path + "\\topfile"
    output_file = output_path + "\\topfile"
    
    def is_float(s):
        """Check if the string s can be converted to a float."""
        try:
            float(s)
            return True
        except ValueError:
            return False


    # Replace all texts
    separation_lines = ['Number of radial and vertical grid points', 'Inside and Outside radius and psi_sep', 'X-coordinates', 'Z-coordinates', 'B_R', 'B_t', 'B_Z', 'psi','you fall asleep']
    separation_index = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            stripped_line = line.strip()
            if not stripped_line:
                # Preserve empty lines
                outfile.write(line)
                continue

            tokens = stripped_line.split()
            if all(is_float(token) for token in tokens):
                # Line is composed entirely of numbers
                outfile.write(line)
            else:
                # Replace non-numeric line with 'X'
                outfile.write(separation_lines[separation_index] + '\n')
                separation_index += 1
        outfile.write(separation_lines[-1])


def beam_length(q_R, q_Z):
    d_length = np.sqrt(np.diff(q_R) ** 2 + np.diff(q_Z) ** 2)
    length = np.cumsum(d_length)
    length = np.insert(length, 0, 0) # ensure that the length of the array is the same as input
    return np.array(length)


def torbeam_in_plasma(data):
    # find the entry and exit point of torbeam in the plasma
    start, end = 0, -1
    for i in range(len(data)):
        if data[i+1] - data[i] == 0 and start !=0 and end == -1:
            end = i
            break 
        if data[i] - data[i-1] != 0 and i != 0 and start == 0:
            start = i
    return start, end


def Te_scaling(filename, scaling_factor: float):
    # Rescale the electron temperature profile
    Te = np.fromfile(filename, dtype=float, sep="   ")
    
    format_string = "{:.12f}"
    for i in range(2, len(Te), 2):
        Te[i] *= scaling_factor
    f = open('Te.dat', 'w')
    f.writelines(['    ', str(int(Te[0])), '\n'])
    for i in range(1, len(Te), 2):
        f.writelines([format_string.format(Te[i]), '  ', str(Te[i + 1]), '\n'])
    f.close()


