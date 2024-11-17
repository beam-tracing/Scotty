import numpy as np
from scotty.fun_general import read_floats_into_list_until


class Torbeam_output:
    # expected output files: t1_LIB.dat, t2_new_LIB.dat, t2_LIB.dat, reflout (if flavour == D)
    def __init__(self, path, flavour):
        # open the files
        self.path = path
        self.flavour = flavour
        self.t1_LIB = np.fromfile(path + '\\t1_LIB.dat',dtype=float, sep='   ')
        self.t2n_LIB = np.fromfile(path + '\\t2_new_LIB.dat',dtype=float, sep='   ')
        self.t2_LIB = np.fromfile(path + '\\t2_LIB.dat',dtype=float, sep='   ')
        if flavour == 'D':
            self.reflout = np.fromfile(path + '\\reflout.dat',dtype=float, sep='   ')
        
        # t1 processing: beam positions
        iend = int(len(self.t1_LIB) / 6)
        t1 = self.t1_LIB.reshape(6, iend, order='F') / 100 # torbeam is in cm, convert to m
        self.q_R = t1[0]
        self.q_Z = t1[1]
        self.upper_q_R = t1[2]
        self.upper_q_Z = t1[3]
        self.lower_q_R = t1[4]
        self.lower_q_Z = t1[5]

        # t2n processing: power deposition and current drive
        npnt = int(len(self.t2n_LIB) / 3)
        self.t2n_LIB = self.t2n_LIB.reshape(3,npnt, order='F')  # rho, power deposition, current drive
        self.rho = self.t2n_LIB[0]
        self.power_deposition = self.t2n_LIB[1]
        self.current_drive = self.t2n_LIB[2]

        # t2 processing: plasa profile data
        nprov = 25 # by convention
        t2 = self.t2_LIB[0: nprov * 3].reshape(3, nprov)
        self.rho_plasma = t2[0]
        self.enclosed_area = t2[1]
        self.enclosed_volume = t2[2]


# work in progress
class y_LIB:
    
    # work in progress
    def __init__(self, path):
        self.y_LIB = np.fromfile(path + '\\y_LIB.dat',dtype=float, sep='   ')
        y = self.y_LIB.reshape(int(len(self.y_LIB) / 19), 19).T
        self.q_R = y[0] / 100
        self.q_zeta = y[1] / 100
        self.q_Z = y[2] / 100
        self.N_R = y[3]
        self.N_zeta = y[4]
        self.N_Z = y[5]
        self.y6 = y[6]
        self.y7 = y[7]
        self.y8 = y[8]
        self.y9 = y[9]
        self.y10 = y[10]
        self.y11 = y[11]
        self.y12 = y[12]
        self.y13 = y[13]
        self.y14 = y[14]
        self.y15 = y[15]
        self.y16 = y[16]
        self.y17 = y[17]
        self.y18 = y[18]
    
    def __repr__(self):
        return f"y_LIB(path={self.path})"