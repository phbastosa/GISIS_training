from sys import argv
from time import time
from numba import njit, prange  

import numpy as np

class BaseModeling():

    def __init__(self) -> None:

        self.parameter_file = argv[1]

    def catch_parameter(self, target : str):

        file = open(self.parameter_file, 'r')
        for line in file.readlines():
            if line[0] != '#':
                splitted = line.split()
                if len(splitted) != 0:
                    if splitted[0] == target: 
                        return splitted[2]

    def import_binary_matrix(self, n1, n2, filename):
        data = np.fromfile(filename, dtype = np.float32, count = n1*n2)        
        return np.reshape(data, [n1, n2], order = "F")

    def set_parameters(self):
        
        self.nx = int(self.catch_parameter("x_samples"))
        self.nz = int(self.catch_parameter("z_samples"))
        self.nt = int(self.catch_parameter("t_samples"))

        self.dx = float(self.catch_parameter("x_spacing"))
        self.dz = float(self.catch_parameter("z_spacing"))
        self.dt = float(self.catch_parameter("t_spacing"))

        self.nabc = int(self.catch_parameter("abc_samples"))
        self.fabc = float(self.catch_parameter("abc_factor"))

        self.fmax = float(self.catch_parameter("max_frequency"))

    def set_Cerjan_abc(self):
        ''' Absorbing Boundary Condition: 

            A nonreflecting boundary condition for discrete acoustic 
            and elastic wave equations; Cerjan et al (1985). '''
        
        self.nxx = self.nx + 2*self.nabc    
        self.nzz = self.nz + 2*self.nabc

        self.damper = np.ones((self.nzz, self.nxx))

        damp_function = np.zeros(self.nabc)

        for k in range(self.nabc):
            damp_function[k] = np.exp(-(self.fabc*(self.nabc-k))**2)
        
        for i in range(self.nzz):
            self.damper[i,:self.nabc] *= damp_function
            self.damper[i,self.nxx-self.nabc:self.nxx] *= damp_function[::-1]

        for j in range(self.nxx):
            self.damper[:self.nabc,j] *= damp_function
            self.damper[self.nzz-self.nabc:self.nzz,j] *= damp_function[::-1]

    def set_geometry(self):
        pass

    def check_geometry_overflow(self):
        pass

    def set_modeling_parameters(self):
        self.set_parameters()
        self.set_Cerjan_abc()
        self.set_geometry()
        self.set_wavelet()
        self.set_models()

        self.check_geometry_overflow()

    def set_wavefield(self):
        raise NotImplementedError("Please Implement this method")

    def set_models(self):
        raise NotImplementedError("Please Implement this method")
    
    def set_wavelet(self):
        raise NotImplementedError("Please Implement this method")


class Scalar(BaseModeling):

    def set_models(self):
        
        self.vp_filename = self.catch_parameter("vp_filename")  
        
        self.vp = self.import_binary_matrix(self.nz, self.nx, self.vp_filename) 

        self.Vp = np.zeros((self.nzz, self.nxx))

        self.Vp[self.nabc:self.nzz-self.nabc,self.nabc:self.nxx-self.nabc] = self.vp.copy()

        for i in range(self.nabc):
            self.Vp[i,self.nabc:self.nxx-self.nabc] = self.vp[0,:]
            self.Vp[self.nzz-i-1,self.nabc:self.nxx-self.nabc] = self.vp[-1,:]

        for i in range(self.nabc):
            self.Vp[:,i] = self.Vp[:,self.nabc]
            self.Vp[:,self.nxx-i-1] = self.Vp[:,self.nxx-self.nabc-1]

    def set_wavelet(self):

        t0 = 2.0*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0

        arg = np.pi*(np.pi*fc*td)**2.0

        self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)        

    def set_wavefield(self):

        self.U_pas = np.zeros_like(self.Vp)
        self.U_pre = np.zeros_like(self.Vp)
        self.U_fut = np.zeros_like(self.Vp)


