import numpy as np

from numba import njit, prange  

from modules.base_modeling import BaseModeling

class Elastic(BaseModeling):

    def __init__(self):
        super().__init__()

        self.name = "Elastic"

    def set_models(self):

        self.vp_filename = self.catch_parameter("vp_filename")[0]  
        self.vs_filename = self.catch_parameter("vs_filename")[0]  
        self.rho_filename = self.catch_parameter("rho_filename")[0]  
        
        self.vp, self.Vp = self.set_generical_model(self.vp_filename)    
        self.vs, self.Vs = self.set_generical_model(self.vs_filename)    
        self.rho, self.Rho = self.set_generical_model(self.rho_filename)    

        self.M = self.Rho*self.Vs**2.0
        self.L = self.Rho*self.Vp**2.0 - 2.0*self.M 

    def set_wavelet(self):

        aux = self.set_generical_wavelet()        

        self.wavelet = np.zeros_like(aux)

        for i in range(self.nt):
            self.wavelet[i] = np.sum(aux[:i+1])        

    def set_wavefield(self):
        
        self.Vx = np.zeros_like(self.Vp) 
        self.Vz = np.zeros_like(self.Vp) 
        self.Txx = np.zeros_like(self.Vp) 
        self.Tzz = np.zeros_like(self.Vp) 
        self.Txz = np.zeros_like(self.Vp) 

    def plot_models(self):

        imgs = [self.vp, self.vs, self.rho]
        pmin = [np.min(self.vp), np.min(self.vs), np.min(self.rho)]
        pmax = [np.max(self.vp), np.max(self.vs), np.max(self.rho)]
        legends = ["P wave velocity [m/s]", "S wave velocity [m/s]", "Density [kg/m³]"]

        self.set_generical_model_plot(imgs, legends, pmin, pmax)

    def apply_wavelet(self):
        
        self.Txx[self.sIdz,self.sIdx] += self.wavelet[self.time_id] / (self.dx*self.dz)
        self.Tzz[self.sIdz,self.sIdx] += self.wavelet[self.time_id] / (self.dx*self.dz)
