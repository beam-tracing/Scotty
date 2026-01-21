import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from scotty.init_bruv import get_parameters_for_Scotty
from scotty.plotting import *
from scotty.geometry import find_nearest, InterpolatedField
from scotty.beam_me_up import make_density_fit, create_magnetic_geometry, beam_me_up

import numpy as np
import pathlib
import os
import copy
import itertools
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#---------------------------------#
# Main page creation
root = tk.Tk()
root.title("Scotty Input")
print("Disclaimer - Entering files will rename them to fit Scotty package")
AVAILABLEPLOTS = []

#---------------------------------#
# Functions
def submit(): # Main beam_me_up() running function

    # File name/path compilation
    mag_path, mag_filename = os.path.split(magnetic_data_path.get())
    spl_mag = mag_filename.split('_')
    if len(spl_mag) != 3 or not spl_mag[0].isdigit() or spl_mag[1:3] != ["equilibrium","data.npz"]:
        # input(f'renaming {mag_filename}') # <- For testing, can be uncommented if needed
        mag_filename = f"{shot.get()}_equilibrium_data.npz"
        os.rename(magnetic_data_path.get(),mag_path+mag_filename)
        magnetic_data_path.set(mag_path+mag_filename)
        B_file_update()

    ne_path, ne_filename = os.path.split(ne_data_path.get())
    if ne_filename[:2] != 'ne':
        # input(f'renaming {ne_filename}') 
        ne_filename = f"ne{ne_filename}"
        os.rename(ne_data_path.get(),ne_filename)
        ne_data_path.set(ne_path+ne_filename)
        ne_file_update()

    # Gather (most) parameters for beam_me_up()
    base_dict = {
        'mode_flag': int(mode_flag.get()),
        'launch_beam_width': float(launch_beam_width.get()),
        'launch_beam_curvature': float(launch_beam_curvature.get()),
        'launch_position': np.array([float(x.strip()) for x in launch_position.get().split(',')]),
        'find_B_method': 'test',
        'shot': int(shot.get()) if shot.get() != "0" else int(mag_filename.split("_")[0]) if mag_filename.split("_")[0].isdigit() else 0,
        'Psi_BC_flag': Psi_BC_flag.get() if Psi_BC_flag.get() != "None" else None,
        'figure_flag': figure_flag.get() == "True",
        'vacuum_propagation_flag': bool(vacuum_propagation_flag.get()),
        'ne_data_path': pathlib.Path(ne_path),
        'magnetic_data_path': pathlib.Path(mag_path),
        'equil_time': float(equil_time.get()),
        'output_filename_suffix': output_file_suffix.get(),
        'quick_run': quick_run.get() == 'True',
        'output_path': pathlib.Path(output_path.get()),
        'input_filename_suffix': ne_filename[2:-4],
        'poloidal_flux_enter': float(poloidal_flux_enter.get()),
        'poloidal_flux_zero_density': float(poloidal_flux_zero_density.get()),
        'delta_R': float(delta_R.get()),
        'delta_Z': float(delta_Z.get()),
        'delta_K_R': float(delta_K_R.get()),
        'delta_K_zeta': float(delta_K_zeta.get()),
        'auto_delta_sign': auto_delta_sign.get() == "True",
        'delta_K_Z': float(delta_K_Z.get()),
        'interp_smoothing': float(interp_smoothing.get()),
        'len_tau': int(len_tau.get()),
        'rtol': float(rtol.get()),
        'atol': float(atol.get())
    }

    # Batch range handling
    batched_vars = {
        'poloidal_launch_angle_Torbeam': poloidal_launch_angle_Torbeam.get(),
        'toroidal_launch_angle_Torbeam': toroidal_launch_angle_Torbeam.get(),
        'launch_freq_GHz': launch_freq_GHz.get(),
    }
    varlbl_lists = []
    rng_lists = []
    for var in batched_vars:
        if '[' in batched_vars[var]: 
            varlbl_lists.append(var)
            rng_lists.append(rmv_dupes(list(map(float,batched_vars[var][1:-1].split(',')))))
        else: 
            base_dict[var] = float(batched_vars[var])

    # Multiple runs
    if varlbl_lists:
        failed_ind = []
        for runind,run in enumerate(itertools.product(*rng_lists)):
            # print(run)
            kwargs_dict = copy.deepcopy(base_dict)
            for ind,lbl in enumerate(varlbl_lists):
                kwargs_dict[lbl] = run[ind]
            try:
                kwargs_dict['output_filename_suffix'] += str(runind +1)
                dt = beam_me_up(**kwargs_dict)
                print(dt)
            except Exception as err:
                failed_ind.append(str(runind+1))
                print(f"Run {runind+1} failed with error - {str(err)}")
        if failed_ind:
            messagebox.showinfo("Info",'Simulations Completed\n\n' \
            f'Simulations which raised errors: {", ".join(failed_ind)}')
        else:
            messagebox.showinfo("Info",'Simulations Completed')

    # Single run
    else:
        try:
            dt = beam_me_up(**base_dict)
            xarray_path.set(f"{base_dict['output_path']}/scotty_output{base_dict['output_filename_suffix']}.h5")
            print(dt)
            messagebox.showinfo("Info",'Simulation Completed')
        except Exception as err:
            messagebox.showinfo("Info",f"Simulation Failed - {err}")
        xr_label.config(text = f"Selected .h5 file: {xarray_path.get()}") 
        xr_updatefunc()

#------------------------------------------------------------------#
# Validation functions for main 'Run' button
def isbatchval(varstr):
    if not varstr.strip(): return False
    if (varstr[0] == '[') != (varstr[-1] == ']'): return False
    if (varstr[0] != '['): return isfloat(varstr)
    else:
        try:
            _ = [float(x.strip()) for x in varstr[1:-1].split(',')]
            return True
        except ValueError:
            return False 

def isfloat(x):
    try:
        float(x)
        return True
    except Exception:
        return False

toroidal_launch_angle_Torbeam = tk.StringVar()
poloidal_launch_angle_Torbeam = tk.StringVar()
launch_freq_GHz = tk.StringVar()

val_torangle = lambda:isbatchval(toroidal_launch_angle_Torbeam.get())
val_polangle = lambda:isbatchval(poloidal_launch_angle_Torbeam.get())

def val_launchpos():
    try:
        pos = [float(x.strip()) for x in launch_position.get().split(',')] 
        return len(pos) == 3
    except ValueError:
        return False

val_launchfreq = lambda:isbatchval(launch_freq_GHz.get())

val_launchbwidth = lambda:isfloat(launch_beam_width.get())
val_launchbcurv = lambda:isfloat(launch_beam_curvature.get())
val_files = lambda:magnetic_data_path.get() and ne_data_path.get()

validators = [val_torangle,val_polangle,val_launchpos,val_launchfreq,val_launchbwidth,val_launchbcurv,val_files]

def validate_launch_all(*args):
    all_valid = all(validator() for validator in validators)
    if all_valid:
        submitbutton.config(state="normal")
        guidance_label.config(text="Ready to run")
    else:
        submitbutton.config(state="disabled")
        guidance_label.config(text="Enter input files and launch parameters in the correct format to enable 'Run' button")

#------------------------------------------------------------------#
# Entered validation functions (used inside the tk.Entry widgets)
def entrfloat(x):
    if x in ['','-']: return True
    try:
        float(x)
        return True
    except Exception:
        return False

def entrpos(x):
    if x.count(',') > 2: 
        return False
    for y in x:
        if (not y.isdigit() and y not in ['','.','-',',']): 
            return False
    return True

def entrbatch(x):
    isv = True
    if any(x.count(char) > 1 for char in "[]"):
        return False
    for y in x:
        try:
            if y not in ['','.','-',',','[',']']:
                int(y)
        except Exception:
            isv = False
            break
    return isv

entrint = lambda x:x.isdigit() or x == ''

# Has to be registered to root to use in the input fields
vadvint = root.register(entrint)
vadvfloat = root.register(entrfloat)
vadvlaunchpos = root.register(entrpos)
vadvbatch = root.register(entrbatch)

#---------------------------------#
# Update available plots factory function
def auto_updatefunc(plotname,criteria,cargsfuncsl):
    def update_availplots(*args):
        mode = criteria(list(cargs() for cargs in cargsfuncsl))
        if mode and plotname not in AVAILABLEPLOTS:
            AVAILABLEPLOTS.append(plotname)
        elif not mode and plotname in AVAILABLEPLOTS:
            AVAILABLEPLOTS.remove(plotname)
        AVAILABLEPLOTS.sort()
        plot_selector_update()
    return update_availplots

#---------------------------------#
# Miscellaneous functions 
# Dropdown function
def autodropdown(pg,options,tvar,row,def_ind): # Uses default col and other params
    ent = ttk.Combobox(pg, textvariable=tvar)
    ent.grid(row=row,column=1,padx=5,pady=5)
    ent['values'] = options  
    if options:
        ent.current(def_ind)
    ent.state(["readonly"])  
    return ent

rmv_dupes = lambda x: [y for ind,y in enumerate(x) if (y not in x[:ind])]
ispresent_func = lambda y: all(x != '' for x in y) 

#---------------------------------#
# File inputs

tk.Label(root, text="Input Files",font=("Arial", 20)).grid(row=0,column=0,columnspan=2, sticky = "n", padx=5, pady=10)

def choose_B_file():
    global magnetic_data_path
    path = filedialog.askopenfilename(
        title="Select a NumPy (.npz) input file",
        filetypes=(("NumPy files", "*.npz"),))
    if path: 
        magnetic_data_path.set(path)
        B_file_update()

B_file_update = lambda:mag_data_l.config(text = f"Magnetic Field File: {magnetic_data_path.get()}")

magnetic_data_path = tk.StringVar()
mag_data_l = tk.Label(root, text=f"Magnetic Field File: {magnetic_data_path.get()}",wraplength=200)
mag_data_l.grid(row=1, column=0, padx=5, pady=5)
B_file_button = tk.Button(root, text="Choose File", command=choose_B_file)
B_file_button.grid(row=2,column=0,pady = 5)

def choose_ne_file():
    global ne_data_path
    path = filedialog.askopenfilename(
    title="Select a .dat input file",
    filetypes=(("Data files", "*.dat"),))
    if path: 
        ne_data_path.set(path)
        ne_file_update()
ne_file_update = lambda:ne_data_l.config(text = f"ne Data File: {ne_data_path.get()}")

ne_data_path = tk.StringVar()
ne_data_l = tk.Label(root, text=f"ne Data File: {ne_data_path.get()}",wraplength=300)
ne_data_l.grid(row=1, column=1, padx=5, pady=5)
ne_file_button = tk.Button(root, text="Choose File", command=choose_ne_file)
ne_file_button.grid(row=2,column=1,pady = 10)

file_launch_separator = ttk.Separator(root, orient="horizontal")
file_launch_separator.grid(row=3, column=0,columnspan=3, sticky="ew", padx=10, pady=5)

#---------------------------------#
# Launch Params 

tk.Label(root, text="Launch Parameters",font=("Arial", 16)).grid(row=4,column=0,columnspan=2, sticky = "n", padx=5, pady=15)

tk.Label(root, text="Toroidal Launch Angle / ˚:").grid(row=5, column=0, sticky="e", padx=5, pady=5)
tk.Label(root, text="Poloidal Launch Angle / ˚:").grid(row=6, column=0, sticky="e", padx=5, pady=5)
tk.Label(root, text="Launch Position / R, ζ, Z:").grid(row=7, column=0, sticky="e", padx=5, pady=5)
tk.Label(root, text="Launch Frequency / GHz:").grid(row=8, column=0, sticky="e", padx=5, pady=5)
tk.Label(root, text="Launch Beam Width / m:").grid(row=9, column=0, sticky="e", padx=5, pady=5)
tk.Label(root, text="Launch Beam Curvature / m⁻¹:").grid(row=10, column=0, sticky="e", padx=5, pady=5)
tk.Label(root, text="Mode Flag:").grid(row=11, column=0, sticky="e", padx=5, pady=5)

launch_position = tk.StringVar()
launch_beam_width = tk.StringVar()
launch_beam_curvature = tk.StringVar()
mode_flag = tk.StringVar()

tk.Entry(root, textvariable=toroidal_launch_angle_Torbeam,validate='all',validatecommand=(vadvbatch,'%P')).grid(row=5, column=1, padx=5, pady=5)
tk.Entry(root, textvariable=poloidal_launch_angle_Torbeam,validate='all',validatecommand=(vadvbatch,'%P')).grid(row=6, column=1, padx=5, pady=5)
tk.Entry(root, textvariable=launch_position,validate='all',validatecommand=(vadvlaunchpos,'%P')).grid(row=7, column=1, padx=5, pady=5)
tk.Entry(root, textvariable=launch_freq_GHz,validate='all',validatecommand=(vadvbatch,'%P')).grid(row=8, column=1, padx=5, pady=5)
tk.Entry(root, textvariable=launch_beam_width,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=9, column=1, padx=5, pady=5)
tk.Entry(root, textvariable=launch_beam_curvature,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=10, column=1, padx=5, pady=5)
mflag_ent = autodropdown(root,['1','-1'],mode_flag,11,0)    

launch_pages_separator = ttk.Separator(root, orient="horizontal")
launch_pages_separator.grid(row=12, column=0,columnspan=3, sticky="ew", padx=10, pady=10)

submitbutton = tk.Button(root, text="Run", command=submit, state="disabled", font=("Arial",20))
submitbutton.grid(row=29, column=0, columnspan=2, pady=10)
guidance_label = tk.Label(root, text="Enter input files and launch parameters in the correct format to enable 'Run' button",font=("Arial", 10),fg='grey')
guidance_label.grid(row=30,column=0,columnspan=2, sticky = "n", padx=5, pady=5)

# Trace all - for enabling/disabling 'Run' button
for var in [poloidal_launch_angle_Torbeam, toroidal_launch_angle_Torbeam, 
            launch_position,launch_freq_GHz,launch_beam_width,
            launch_beam_curvature,mode_flag,ne_data_path,
            magnetic_data_path]:
    var.trace_add("write", validate_launch_all)

#---------------------------------#
# Info Labels
def show_info(text):
    messagebox.showinfo("Info",text)

def create_infolabel(pg,text,r,c,padx = 10,pady = 5): # Creates info label widget
    btn_frame = tk.Frame(pg, bg="white")
    info_btn = tk.Label(
        btn_frame,
        text="i",fg="black",bg="#E6E6E6",font=("Roboto", 9, "bold"),
        width=2,height=1,relief="flat",cursor="hand2",bd=0
        )
    info_btn.bind("<Button-1>", lambda e: show_info(text))
    info_btn.bind("<Enter>", lambda e: info_btn.config(bg="#C2C2C2"))
    info_btn.bind("<Leave>", lambda e: info_btn.config(bg="#E6E6E6"))
    info_btn.pack()
    btn_frame.grid(row = r, column= c,padx=padx, pady=pady)
    return btn_frame

file_input_info = create_infolabel(
    root,
    "Files to be used in beam_me_up(), magnetic field Numpy file (.npz) "\
    "and equilibrium electron density data file (.dat) respectively\n\n" \
    "Files will be renamed in the case that they are not in the correct naming convention "
    "([shot]_equilibrium_data.npz / ne[suffix].dat)",
    0,2
)
launch_param_info = create_infolabel(
    root,
    "Configure basic beam and diagnostic parameters for your simulation.\n\n"\
    "Valid entries for all launch parameters are required to run beam_me_up()",
    4,2
)
tor_info = create_infolabel(
    root,
    "Launch angle in the toroidal plane/top-down of the plasma in degrees.\n\n" \
    "A larger value will lower the X coordinate of the entry point, " \
    "while a value towards the negatives will raise the X coordinate of the entry point.\n\n" \
    "Multiple can be entered in a [value,value] format.",
    5,2
    )
pol_info = create_infolabel(
    root,
    "Launch angle in the poloidal plane/cross-section of plasma in degrees.\n\n" \
    "A larger value will lower the Z coordinate or height of the entry point, " \
    "while a value towards the negatives will raise the Z coordinate of the entry point.\n\n" \
    "Multiple can be entered in a [value,value] format.",
    6,2
    )
launch_pos_info = create_infolabel(
    root,
    "Initial launch position of the beam in cylindrical coordinates.\n\n" \
    "Must contain 3 comma-separated values to be valid.",
    7,2
    )
launch_freq_info = create_infolabel(
    root,
    "Frequency of the launch beam in GHz.\n\n" \
    "Multiple inputs can be entered in a [value,value] format.",
    8,2
    )
launch_width_info = create_infolabel(
    root,
    "Width of the beam at launch in m.\n\nMust be above 0.",
    9,2
    )
launch_curv_info = create_infolabel(
    root,
    "Curvature of the beam at launch in m.",
    10,2
)
mode_flag_info = create_infolabel(
    root,
    "Electromagnetic wave mode of the beam, " \
    "1 corresponds to O-Mode, -1 corresponds to X-Mode.",
    11,2
)

#---------------------------------#
# Advanced parameters
advanced_window = None

# Advanced parameters variable init
shot = tk.StringVar()
shot.set('0')
equil_time = tk.StringVar()
equil_time.set('0.0')
len_tau = tk.StringVar()
len_tau.set('102')
poloidal_flux_enter = tk.StringVar()
poloidal_flux_enter.set('1.0')
poloidal_flux_zero_density = tk.StringVar()
poloidal_flux_zero_density.set('1.01')
Psi_BC_flag = tk.StringVar()
figure_flag = tk.StringVar()
vacuum_propagation_flag = tk.StringVar()
quick_run = tk.StringVar()
output_file_suffix = tk.StringVar()
output_path = tk.StringVar() 

# Functions
def advancedwindow_close():
    global advanced_window
    advanced_window.destroy()
    advanced_window = None

def open_advanced():
    global advanced_window
    
    if advanced_window is not None and advanced_window.winfo_exists():
        advanced_window.lift()
        advanced_window.focus_force()
        return

    advanced_window = tk.Toplevel(root)
    advanced_window.protocol("WM_DELETE_WINDOW", advancedwindow_close)

    tk.Label(advanced_window, text="Advanced Parameters",font=("Arial", 20)).grid(row=0,column=0,columnspan=2, sticky = "n", padx=5, pady=10)

    # Frames + Separators
    adv_tbeam_separator = ttk.Separator(advanced_window, orient="horizontal")
    adv_tbeam_separator.grid(row=1, column=0,columnspan=3, sticky="ew", padx=10, pady=5)
    adv_topframe = tk.Frame(advanced_window)
    adv_topframe.grid(row=2, column=0, sticky="nsew")
    adv_beamoutput_separator = ttk.Separator(advanced_window, orient="horizontal")
    adv_beamoutput_separator.grid(row=3, column=0,columnspan=3, sticky="ew", padx=10, pady=5)
    adv_bottomframe = tk.Frame(advanced_window)
    adv_bottomframe.grid(row=4, column=0, sticky="nsew")

    # Top Frame - Beam Behaviour
    tk.Label(adv_topframe, text="Beam Behaviour",font=("Arial", 16)).grid(row=0,column=0,columnspan=2, sticky = "n", padx=5, pady=15)

    tk.Label(adv_topframe, text="Time of Equilibrium / s:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    tk.Label(adv_topframe, text="Poloidal Flux Enter:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    tk.Label(adv_topframe, text="Poloidal Flux Zero Density:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
    tk.Label(adv_topframe, text="Psi-BC Flag:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
    tk.Label(adv_topframe, text="Vacuum Propagation Flag:").grid(row=5, column=0, sticky="e", padx=5, pady=5)

    tk.Entry(adv_topframe, textvariable=equil_time,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=1, column=1, padx=5, pady=5)
    tk.Entry(adv_topframe, textvariable=poloidal_flux_enter,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=2, column=1, padx=5, pady=5)
    tk.Entry(adv_topframe, textvariable=poloidal_flux_zero_density,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=3, column=1, padx=5, pady=5)
    psiflag_ent = autodropdown(adv_topframe,['None','discontinuous','continuous'],Psi_BC_flag,4,0) 
    vacpropflag_ent = autodropdown(adv_topframe,['True','False'],vacuum_propagation_flag,5,0)

    equil_time_info = create_infolabel(
        adv_topframe,
        "Time of Equilibrium taken from magnetic field file.\n\n" \
        "Different times will result in a different state of the magnetic field.", 
        1,2
        )
    polflux_enter_info = create_infolabel(
        adv_topframe,
        "Normalised poloidal flux label of plasma boundary.", 
        2,2
        )
    polflux_zero_info = create_infolabel(
        adv_topframe,
        "Poloidal flux label while density is 0.\n\n" \
        "Cannot be lower than poloidal flux enter.", 
        3,2
        )
    psibc_info = create_infolabel(
        adv_topframe,
        "How boundary conditions (BCs) are applied at the plasma-vacuum boundary.\n\n" \
        "None - No special treatment at plasma-vac boundary.\n\n" \
        "Continuous - Apply BCs for continuous electron density but discontinuous gradient.\n\n" \
        "Discontinuous - Apply BCs for discontinuous electron density.",
        4,2
        )
    vacprop_flag_info = create_infolabel(
        adv_topframe,
        "Determines if solver runs from launch position (True)"
        " or uses analytical vacuum propagation (False)",
        5,2
        )
    
    # Bottom Frame - Output Settings
    tk.Label(adv_bottomframe, text="Output Settings",font=("Arial", 16)).grid(row=0,column=0,columnspan=2, sticky = "n", padx=5, pady=15)

    tk.Label(adv_bottomframe, text="Shot No.:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    tk.Label(adv_bottomframe, text="Plot Points (tau_values):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    tk.Label(adv_bottomframe, text="Generate Figures:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
    tk.Label(adv_bottomframe, text="Quick Run:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
    tk.Label(adv_bottomframe, text="Output Suffix:").grid(row=5, column=0, sticky="e", padx=5, pady=5)

    tk.Entry(adv_bottomframe, textvariable=shot,validate='all',validatecommand=(vadvint,'%P')).grid(row=1, column=1, padx=5, pady=5)
    tk.Entry(adv_bottomframe, textvariable=len_tau,validate='all',validatecommand=(vadvint,'%P')).grid(row=2, column=1, padx=5, pady=5)
    figflag_ent = autodropdown(adv_bottomframe,['True','False'],figure_flag,3,0)
    quickrun_ent = autodropdown(adv_bottomframe,['True','False'],quick_run,4,1)
    tk.Entry(adv_bottomframe, textvariable=output_file_suffix).grid(row=5, column=1, padx=5, pady=5)

    def choose_outputfolder():
        global output_path
        path = filedialog.askdirectory(title="Select a folder")
        if path: 
            output_path.set(path)
            output_path_l.config(text = f"Output Path: {output_path.get()}")

    output_path_l = tk.Label(adv_bottomframe, text=f"Output Path: {output_path.get()}",wraplength=200)
    output_path_l.grid(row=6, column=0, padx=5, pady=5,sticky='e')
    output_path_button = tk.Button(adv_bottomframe, text="Choose Folder", command=choose_outputfolder)
    output_path_button.grid(row=6,column=1,pady = 5)

    shot_no_info = create_infolabel(
        adv_bottomframe,
        "Experimental Shot number, used only for file labeling.\n\n" \
        "Leave at 0 if magnetic data file is in valid format.",
        1,2
        )
    
    tau_vals_info = create_infolabel(
        adv_bottomframe,
        "Number of tau points to use in the simulation along the beam path.\n\n" \
        "More points will result in more detail but longer processing time", 
        2,2
        )
    figflag_info = create_infolabel(
        adv_bottomframe,
        "Decides if figures are generated after beam_me_up() has finished running.\n\n" \
        "Graphs of dispersion relation and poloidal beam path would be saved "
        "in the same output path as the .h5 file, specified in the entry below",
        3,2
        )
    quickrun_info = create_infolabel(
        adv_bottomframe,
        "Runs only the ray tracer and gets an analytic estimate of the K cut-off location if enabled.",
        4,2
        )
    outputsuffix_info = create_infolabel(
        adv_bottomframe,
        "Determines the output suffix of the .h5 (and graphs if enabled) files.\n\n" \
        "The index of the run will be automatically appended to the suffix if multiple simulations are run at the same time.",
        5,2
        )
    outputfilepath_info = create_infolabel(
        adv_bottomframe,
        "Determines the filepath that the resultant files are saved to.",
        6,2
        )

advpg_button = tk.Button(root, text="Advanced Parameters", command=open_advanced,width=20,font=("Arial",14))
advpg_button.grid(row=27, column=0,columnspan=1, pady=10)

#---------------------------------#
# Solver parameters
solverwindow = None

auto_delta_sign = tk.StringVar()
delta_R = tk.StringVar()
delta_R.set('-0.0001')
delta_Z = tk.StringVar()
delta_Z.set('0.0001')
delta_K_R = tk.StringVar()
delta_K_R.set('0.1')
delta_K_zeta = tk.StringVar()
delta_K_zeta.set('0.1')
delta_K_Z = tk.StringVar()
delta_K_Z.set('0.1')
interp_smoothing = tk.StringVar()
interp_smoothing.set('0')
atol = tk.StringVar()
atol.set('0.000001')
rtol = tk.StringVar()
rtol.set('0.001')

def solverwindow_close():
    global solverwindow
    solverwindow.destroy()
    solverwindow = None

def open_solver_params():
    global solverwindow

    if solverwindow is not None and solverwindow.winfo_exists():
        solverwindow.lift()
        solverwindow.focus_force()
        return

    solverwindow = tk.Toplevel(root)
    solverwindow.protocol("WM_DELETE_WINDOW", solverwindow_close)

    tk.Label(solverwindow, text="Solver Parameters",font=("Arial", 15)).grid(row=1,column=0,columnspan=2, sticky = "n", padx=5, pady=10)

    tk.Label(solverwindow, text="Auto Delta Sign:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    tk.Label(solverwindow, text="Delta R / m:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
    tk.Label(solverwindow, text="Delta Z / m:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
    tk.Label(solverwindow, text="Delta K_R / m⁻¹:").grid(row=5, column=0, sticky="e", padx=5, pady=5)
    tk.Label(solverwindow, text="Delta K_ζ:").grid(row=6, column=0, sticky="e", padx=5, pady=5)
    tk.Label(solverwindow, text="Delta K_Z / m⁻¹:").grid(row=7, column=0, sticky="e", padx=5, pady=5)
    tk.Label(solverwindow, text="Interpolation Smoothing:").grid(row=8, column=0, sticky="e", padx=5, pady=5)
    tk.Label(solverwindow, text="atol:").grid(row=9, column=0, sticky="e", padx=5, pady=5)
    tk.Label(solverwindow, text="rtol:").grid(row=10, column=0, sticky="e", padx=5, pady=5)

    delta_sign_dropdown = autodropdown(solverwindow,['True','False'],auto_delta_sign,2,0)
    tk.Entry(solverwindow, textvariable=delta_R,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=3, column=1, padx=5, pady=5)
    tk.Entry(solverwindow, textvariable=delta_Z,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=4, column=1, padx=5, pady=5)
    tk.Entry(solverwindow, textvariable=delta_K_R,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=5, column=1, padx=5, pady=5)
    tk.Entry(solverwindow, textvariable=delta_K_zeta,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=6, column=1, padx=5, pady=5)
    tk.Entry(solverwindow, textvariable=delta_K_Z,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=7, column=1, padx=5, pady=5)
    tk.Entry(solverwindow, textvariable=interp_smoothing,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=8, column=1, padx=5, pady=5)
    tk.Entry(solverwindow, textvariable=atol,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=9, column=1, padx=5, pady=5)
    tk.Entry(solverwindow, textvariable=rtol,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=10, column=1, padx=5, pady=5)

    auto_delta_sign_info = create_infolabel(
        solverwindow,
        "Ensures that forward difference is always in negative poloidal" \
        " flux gradient direction (into the plasma).",
        2,2
    )
    delta_R_info = create_infolabel(
        solverwindow,
        "Finite difference spacing to use for R",
        3,2
    )
    delta_Z_info = create_infolabel(
        solverwindow,
        "Finite difference spacing to use for Z",
        4,2
    )
    delta_K_R_info = create_infolabel(
        solverwindow,
        "Finite difference spacing to use for K_R",
        5,2
    )
    delta_K_zeta_info = create_infolabel(
        solverwindow,
        "Finite difference spacing to use for K_ζ",
        6,2
    )
    delta_K_Z_info = create_infolabel(
        solverwindow,
        "Finite difference spacing to use for K_Z",
        7,2
    )
    interp_smoothing_info = create_infolabel(
        solverwindow,
        "Smoothing factor for interpolation of plasma profiles", 
        8,2
    )
    atol_info = create_infolabel(
        solverwindow,
        "Absolute tolerance for ODE solver.",
        9,2
    )
    rtol_info = create_infolabel(
        solverwindow,
        "Relative tolerance for ODE solver.", 
        10,2
    )


solverpg_button = tk.Button(root, text="Solver Parameters", command=open_solver_params)
solverpg_button.grid(row=28, column=0, pady=10)

#---------------------------------#
# Diagnostic implementation
diagnosticwindow = None

# Variables
diagnostic = tk.StringVar()
d_launch_freq = tk.StringVar()
DIAGNOSTICS = ['DBS_NSTX_MAST','DBS_SWIP_MAST-U','DBS_UCLA_MAST-U','DBS_synthetic']
diagnostic_set_button = None

def diagnostic_close():
    global diagnosticwindow
    diagnosticwindow.destroy()
    diagnosticwindow = None

# Validation function diagnostics
def validate_diagnostic(*args):
    global diagnostic_set_button
    if (isfloat(d_launch_freq.get()) and diagnostic.get() in DIAGNOSTICS) or diagnostic.get() == "DBS_synthetic":
        diagnostic_set_button.config(state = "normal")
    else:
        diagnostic_set_button.config(state = "disabled")

# Function to set diagsnotics
def set_diagnostic():
    if diagnostic.get() not in DIAGNOSTICS: return
    n_params = get_parameters_for_Scotty(diagnostic=diagnostic.get(),launch_freq_GHz=float(d_launch_freq.get() if d_launch_freq.get() else 0 ))
    launch_freq_GHz.set(n_params["launch_freq_GHz"])
    launch_beam_width.set(n_params["launch_beam_width"])
    launch_beam_curvature.set(n_params["launch_beam_curvature"])
    launch_position.set(','.join(str(coord) for coord in n_params["launch_position"]))
    Psi_BC_flag.set(n_params["Psi_BC_flag"])
    figure_flag.set(str(n_params["figure_flag"]))
    vacuum_propagation_flag.set(str(n_params["vacuum_propagation_flag"]))
    if diagnostic.get() == "DBS_synthetic":
        poloidal_launch_angle_Torbeam.set(n_params["poloidal_launch_angle_Torbeam"])
        toroidal_launch_angle_Torbeam.set(n_params["toroidal_launch_angle_Torbeam"]) 
        mode_flag.set(n_params["mode_flag"])
        poloidal_flux_zero_density.set(n_params["poloidal_flux_zero_density"])
        poloidal_flux_enter.set(n_params["poloidal_flux_enter"])

def open_diagnostics():
    global diagnosticwindow,diagnostic_set_button
    
    if diagnosticwindow is not None and diagnosticwindow.winfo_exists():
        diagnosticwindow.lift()
        diagnosticwindow.focus_force()
        return
    
    diagnosticwindow = tk.Toplevel(root)
    diagnosticwindow.protocol("WM_DELETE_WINDOW",diagnostic_close)

    tk.Label(diagnosticwindow, text="Diagnostic Presets",font=("Arial", 16)).grid(row=0,column=0,columnspan=2, sticky = "n", padx=5, pady=15)

    ent = ttk.Combobox(diagnosticwindow, textvariable=diagnostic,height=20,font=("Arial",14))
    ent.grid(row=2,column=0,columnspan=2,padx=5,pady=10)
    ent['values'] = DIAGNOSTICS
    ent.state(["readonly"])  

    tk.Label(diagnosticwindow, text="Launch Frequency / GHz:").grid(row=3,column=0, sticky = "n", padx=5, pady=5)
    tk.Entry(diagnosticwindow, textvariable=d_launch_freq,validate='all',validatecommand=(vadvfloat,'%P')).grid(row=3, column=1, padx=5, pady=5)

    diagnostic_set_button = tk.Button(diagnosticwindow, text="Set Diagnostic", command=set_diagnostic,state ="disabled")
    diagnostic_set_button.grid(row=4,column=0,columnspan=2,pady=10)

    diagnostic_info = create_infolabel(
        diagnosticwindow,
        "Set diagnostic presets from Scotty.\n\n" \
        "Will replace all affected variables (launch beam width, curvature, position) even if filled up.\n\n" \
        "Option DBS_synthetic does not require frequency input and will replace almost all inputs.",
        0,2
    )

    d_freq_info = create_infolabel(
        diagnosticwindow,
        "Frequency of the launch beam in GHz, to be used to calculate values for diagnostic.\n\n" \
        "The input in the diagnostic page is not synced to the input in the main page.\n\n" \
        "Batch variables do not work in this field.",
        3,2
    )

for var in [diagnostic,d_launch_freq]:
    var.trace_add("write",validate_diagnostic)

diagnospg_button = tk.Button(root, text="Diagnostic Presets", command=open_diagnostics)
diagnospg_button.grid(row=28, column=1, pady=10)

#---------------------------------#
# Notes page - for additional things the user might want to know
# - can delete if needed
notes = None

def notes_close():
    global notes
    notes.destroy()
    notes = None

def open_notes():
    global notes
    
    if notes is not None and notes.winfo_exists():
        notes.lift()
        notes.focus_force()
        return
    
    notes = tk.Toplevel(root)
    notes.title("Notes")
    notes.protocol("WM_DELETE_WINDOW",notes_close)

    tk.Label(notes, 
             text="In the case that magnetic data/ne data files are not in naming conventions suitable for Scotty to use, files will be automatically renamed",
             wraplength=500).grid(row=0,column=0,columnspan=2, padx=5, pady=5)
    tk.Label(notes, 
             text="Only .npz files are able to be used as input files currently",
             wraplength=500).grid(row=1,column=0,columnspan=2, padx=5, pady=5)
    tk.Label(notes, 
             text="For 'Enter' button to function, all launch Parameters must be filled with a valid format, especially Launch Position which must be in a 'x,y,z' format where x,y and z are all floats",
             wraplength=500).grid(row=2,column=0,columnspan=2, padx=5, pady=5)
    tk.Label(notes, 
             text="Shot number will be automatically derived from the magnetic data file name if shot is left at 0 and magnetic data file is in correct naming convention for Scotty",
             wraplength=500).grid(row=3,column=0,columnspan=2, padx=5, pady=5)
    tk.Label(notes, 
             text="Sweep variarbles include the launch angles and launch frequency, formatted in [n,n,n] where n is any float value. Do note that duplicate values are automatically removed",
             wraplength=500).grid(row=4,column=0,columnspan=2, padx=5, pady=5)

notes_button = tk.Button(root, text="Notes", command=open_notes)
notes_button.grid(row=31, column=0,columnspan=2, pady=5)

#---------------------------------#
# Plot Functions unrelated to GUI
ne_data_path.trace_add('write', auto_updatefunc('Density Fit Graph (Simple)',ispresent_func,[ne_data_path.get]))

# Magnetic field surface + density heatmap traces
mfile_vars = [equil_time,magnetic_data_path,interp_smoothing]
for var in mfile_vars:
    var.trace_add('write',auto_updatefunc(
        "Magnetic Flux Surface",ispresent_func,[v.get for v in mfile_vars]
    ))
    var.trace_add('write',auto_updatefunc(
        "Poloidal Cross-section",ispresent_func,[v.get for v in mfile_vars]
    ))
# Density Fit Heatmap hidden    
#     var.trace_add('write',auto_updatefunc(
#         "Density Fit Heatmap", ispresent_func,[v.get for v in mfile_vars + [ne_data_path]]
#     ))
# ne_data_path.trace_add('write', auto_updatefunc('Density Fit Heatmap',ispresent_func,[v.get for v in mfile_vars + [ne_data_path]]))

# Full input graph traces
fullinp_vars = [poloidal_launch_angle_Torbeam,toroidal_launch_angle_Torbeam,
            launch_position,poloidal_flux_enter,magnetic_data_path,equil_time,interp_smoothing]
for var in fullinp_vars:
    var.trace_add('write',auto_updatefunc("Full Input Graph",ispresent_func,[v.get for v in fullinp_vars]))

# Function that loads magnetic field object
# - can modify to use create_magnetic_geometry()
def load_magnetic_field():
    with np.load(magnetic_data_path.get()) as loadfile:
        time_EFIT = loadfile["time_EFIT"]
        t_idx = find_nearest(time_EFIT, float(equil_time.get()))
        return InterpolatedField(
            R_grid=loadfile["R_EFIT"],
            Z_grid=loadfile["Z_EFIT"],
            psi=loadfile["poloidalFlux_grid"][t_idx, :, :],
            B_T=loadfile["Bphi_grid"][t_idx, :, :],
            B_R=loadfile["Br_grid"][t_idx, :, :],
            B_Z=loadfile["Bz_grid"][t_idx, :, :],
            interp_order=5,
            interp_smoothing=float(interp_smoothing.get()),
        )

#---------------------------------#
# Plots
def magnetic_flux_surface_3D(fig):
    ax = fig.add_subplot(111, projection="3d")  
    field = load_magnetic_field()
    return plot_flux_surface_3D(field,1.0,ax)

def density_fit_launch(fig):
    ax = fig.add_subplot(111) 
    ne_data = np.fromfile(ne_data_path.get(), dtype=float, sep="   ")
    ne_data_density_array = ne_data[2::2]
    ne_data_radialcoord_array = ne_data[1::2]
    ax.scatter(ne_data_radialcoord_array, ne_data_density_array,color='blue', s=20)
    ax.set_xlabel("Rho")
    ax.set_ylabel("Density (10^19 m^-3)")
    ax.set_title(f"Density Profile")
    ax.grid(True)
    return ax

def density_fit_heatmap(fig):
    ax = fig.add_subplot(111) 
    field = load_magnetic_field()
    densfit = make_density_fit("smoothing-spline-file",1.0,[ne_data_path.get()],filename=ne_data_path.get())
    rho = np.sqrt(field.poloidalFlux_grid.T)
    density_grid = densfit(rho)
    density_grid[density_grid <= 0] = np.nan
    # print(density_grid)
    m = ax.pcolormesh(field.R_coord, field.Z_coord, density_grid, shading='auto', cmap='plasma')
    cbar = fig.colorbar(m, ax=ax)
    cbar.set_label("Density (10^19 m^-3)")
    ax.set_xlabel("R (m)") 
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Density Profile (Heatmap)")
    ax.grid(True)    
    return ax

def plot_all_inp(fig):
    ax = fig.add_subplot(111, projection="3d")  
    field = load_magnetic_field()
    try:
        lpos = np.array([float(x.strip()) for x in launch_position.get().split(',')])
        lpos_cart = cylindrical_to_cartesian(*lpos)
    except Exception as err:
        messagebox.showinfo("Info",'Ensure launch position is in correct format of {R, ζ, Z}')
        print(err)
        pass
    return plot_all_the_things(
        field=field,
        launch_position=lpos_cart,
        toroidal_launch_angle=np.deg2rad(float(toroidal_launch_angle_Torbeam.get())),
        poloidal_launch_angle=np.deg2rad(float(poloidal_launch_angle_Torbeam.get())),
        poloidal_flux_enter=float(poloidal_flux_enter.get()),
        ax = ax)

def poloidal_crosssection(fig):
    ax = fig.add_subplot(111) 
    field = load_magnetic_field()
    R_min = np.min(field.R_coord)
    R_max = np.max(field.R_coord)
    Z_min = np.min(field.Z_coord)
    Z_max = np.max(field.Z_coord)

    ax.hlines([Z_min, Z_max], R_min, R_max, color="lightgrey", linestyle="dashed")
    ax.vlines([R_min, R_max], Z_min, Z_max, color="lightgrey", linestyle="dashed")
    bounds = np.array((R_min, R_max, Z_min, Z_max))
    ax.axis(bounds * 1.1)
    ax.axis("equal")
    cax = ax.contour(
            field.R_coord,
            field.Z_coord,
            (field.poloidalFlux_grid).T,
            levels=np.linspace(0, 1, 11, endpoint=False),
            cmap="plasma_r",
        )
    ax.clabel(cax, inline=True)
    ax.contour(
        field.R_coord,
        field.Z_coord,
        (field.poloidalFlux_grid).T,
        levels=[1.0],
        colors="darkgrey",
        linestyles="dashed",
    )
    dashed_line = mlines.Line2D(
        [], [], color="black", linestyle="dashed", label=r"$\psi = 1$"
    )
    ax.legend(handles=[dashed_line])
    return ax

# Factory function for ALL plotting functions that take in datatree as input
def auto_dtplotfunc(func,modes = [],*args): 
    def plot_output(fig):
        global slider
        if 'grid' in modes:
            ax = fig.subplots(*args) 
        else:
            ax = fig.add_subplot(111) 
        with xr.open_datatree(xarray_path.get(), engine="h5netcdf") as dt:
            # print(dt)
            if "nan_ax" in modes:
                ax = func(dt)
            elif "analysis" in modes:
                ax = func(dt["analysis"],ax=ax)
            elif "3D_plot" in modes:
                ax,slider = func(dt,True,True,True,True,True)
            else:
                ax = func(dt,ax=ax)
        return ax
    return plot_output 

# Global dictionaries of plots - for code reference
INPUT_PLOT_FUNCTIONS = {
    "Magnetic Flux Surface": magnetic_flux_surface_3D,
    "Density Fit Graph (Simple)": density_fit_launch,
    "Density Fit Heatmap": density_fit_heatmap,
    "Full Input Graph": plot_all_inp,
    "Poloidal Cross-section": poloidal_crosssection
}

DT_PLOT_FUNCTIONS = {
    "Poloidal Beam Path": auto_dtplotfunc(plot_poloidal_beam_path),
    "Dispersion Relation": auto_dtplotfunc(plot_dispersion_relation,modes = ["analysis"]),
    "Psi Plots": auto_dtplotfunc(plot_psi,modes = ["nan_ax"]),
    "Toroidal Beam Path": auto_dtplotfunc(plot_toroidal_beam_path),
    "Instrumentation Functions":auto_dtplotfunc(plot_instrumentation_functions,["grid"],3,2),
    "Widths":auto_dtplotfunc(plot_widths),
    "3D Beam Profile": auto_dtplotfunc(plot_3D_beam_profile_3D_plotting,["3D_plot"])
}

# Plot Categories
basic_plots = ["Full Input Graph", "Poloidal Cross-section", "Poloidal Beam Path", "Toroidal Beam Path","3D Beam Profile"] #???
input_plots = ["Magnetic Flux Surface", "Density Fit Graph (Simple)", "Full Input Graph", "Poloidal Cross-section"]
output_plots = ["Poloidal Beam Path", "Psi Plots","Toroidal Beam Path","Widths","3D Beam Profile"]
debugging_plots = ["Dispersion Relation","Instrumentation Functions"]
CATEGORIES = {
    "Basic Plots": basic_plots,
    "Input Plots": input_plots,
    "Output Plots": output_plots,
    "Debugging Plots": debugging_plots
}

#---------------------------------#
# Plotting window
plot_window = None
selcat = tk.StringVar()
selplot = tk.StringVar()
xarray_path = tk.StringVar()
plot_selector = None

def plots_close():
    global plot_window
    plot_window.destroy()
    plot_window = None

# Updates both button and plot selector dropdown
def plot_selector_update(): 
    global plot_selector,toplot_button
    try:
        if plot_selector and plot_selector.winfo_exists():
            currplots = [plot for plot in AVAILABLEPLOTS if plot in CATEGORIES[selcat.get()]]
            plot_selector['values'] = currplots
            if currplots:
                plot_selector.current(0)
                plot_selector.config(state="readonly")  
            else:
                plot_selector.set('')
                plot_selector.config(state="disabled")
        if toplot_button and toplot_button.winfo_exists():
            toplot_button.config(state="normal" if AVAILABLEPLOTS else "disabled")
    except NameError:
        pass

def update_plot(): 
    chosenplot = plot_selector.get()
    plt.clf()
    fig = plt.gcf()  # get current figure
    if chosenplot in INPUT_PLOT_FUNCTIONS: 
        INPUT_PLOT_FUNCTIONS[chosenplot](fig)
    elif chosenplot in DT_PLOT_FUNCTIONS:
        DT_PLOT_FUNCTIONS[chosenplot](fig)
    else:
        print("Error - Plot not found")
    plt.show(block=False)

def choose_h5_file():
    global xr_label
    path = filedialog.askopenfilename(
        title = "Select a .h5 datatree file",
        filetypes=(("HDF5 files", "*.h5"),)
    )
    if path:
        xarray_path.set(path)
        print(xarray_path.get())
        xr_label.config(text = f"Selected .h5 file: {xarray_path.get()}")

def xr_updatefunc(*args):
    global toplot_button
    mode = xarray_path.get() != ''
    for plotname in DT_PLOT_FUNCTIONS:
        if mode and plotname not in AVAILABLEPLOTS:
            AVAILABLEPLOTS.append(plotname)
        elif not mode and plotname in AVAILABLEPLOTS:
            AVAILABLEPLOTS.remove(plotname)
    AVAILABLEPLOTS.sort()
    plot_selector_update()

def open_plot_window():
    global plot_window, plot_selector,xr_label, toplot_button
    
    if plot_window is not None and plot_window.winfo_exists():
        plot_window.lift()
        plot_window.focus_force()
        return
    
    plot_window = tk.Toplevel(root)
    plot_window.title("Plotting")
    plot_window.protocol("WM_DELETE_WINDOW",plots_close)

    tk.Label(plot_window, text="Select Plots",font=("Arial", 16)).grid(row=0,column=0,columnspan=2, sticky = "nsew", padx=5, pady=15)

    tk.Label(plot_window, text="Category: ").grid(row=1,column=0, sticky = "e", padx=5, pady=5)
    cat_selector = autodropdown(plot_window,["Basic Plots","Input Plots","Output Plots","Debugging Plots"],selcat,1,0) 

    plot_selector = autodropdown(plot_window,AVAILABLEPLOTS,selplot,2,0)
    plot_selector.grid(row = 2, column = 0, sticky= "nsew", columnspan= 2, pady= 10)  
    plot_selector_update()
    xr_label = tk.Label(plot_window, text=f"Selected .h5 file: {xarray_path.get()}", wraplength=200)
    xr_label.grid(row=3,column=0, sticky = "nsew", padx=5, pady=10)

    sel_xr_button = tk.Button(plot_window, text="Select File", command=choose_h5_file,font=("Arial",12))
    sel_xr_button.grid(row=3, column=1, pady=10)
    toplot_button = tk.Button(plot_window, text="Plot Selected Graph", command=update_plot,width=15,font=("Arial",14),state='disabled')
    toplot_button.grid(row=4, column=0,columnspan=2, pady=10)

    plot_info = create_infolabel(
        plot_window,
        "Plots graphs based on inputs and outputs given.\n\n" \
        "Choices for each graph will only be available for selection when all required input parameters / .h5 output file are filled up.\n\n" \
        "When running individual runs, will automatically select the .h5 output file of previous run.",
        0,2
    )
    selcat.trace_add('write',lambda *_: plot_selector_update())
    xarray_path.trace_add('write',xr_updatefunc)
    xr_updatefunc()

plots_button = tk.Button(root, text="Plotting", command=open_plot_window,width=15,font=("Arial",14))
plots_button.grid(row=27, column=1,columnspan=1, pady=10)

# gui debugging
# for widget in root.winfo_children():
#     try:
#         text = widget.cget("text")
#     except:
#         text = None
#     grid_info = widget.grid_info()
#     print('-----')
#     print(widget)
#     print(widget.winfo_class())
#     print(text)
#     print(grid_info["column"],grid_info["row"])

#---------------------------------#
# Main loop
root.mainloop()
