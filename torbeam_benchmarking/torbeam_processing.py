import numpy as np
from scotty.fun_general import read_floats_into_list_until


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
    input_file = input_path + "\\topfile"
    output_file = output_path + "\\topfile"
    
    def is_float(s):
        """Check if the string s can be converted to a float."""
        try:
            float(s)
            return True
        except ValueError:
            return False


    # Replace all texts with Scotty format
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


