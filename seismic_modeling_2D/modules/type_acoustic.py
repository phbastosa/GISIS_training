import numpy as np

from numba import njit, prange  

from modules.base_modeling import BaseModeling

class Acoustic(BaseModeling):

    def __init__(self):
        super().__init__()

        self.name = "Acoustic"

    def set_models(self):

        self.vp_filename = self.catch_parameter("vp_filename")[0]  
        self.rho_filename = self.catch_parameter("rho_filename")[0]  
        
        # self.vp, self.Vp = self.set_generical_model(self.vp_filename)    
        # self.rho, self.Rho = self.set_generical_model(self.rho_filename)    

        self.vp = 1500.0*np.ones((self.nz, self.nx))
        self.Vp = 1500.0*np.ones((self.nzz, self.nxx))

        self.rho = 1000.0*np.ones((self.nz, self.nx))
        self.Rho = 1000.0*np.ones((self.nzz, self.nxx))

        self.Bx = 1.0 / self.Rho
        self.Bz = 1.0 / self.Rho

        self.K = self.Rho*self.Vp**2.0
        
        self.Bx[:-1,:] = 0.5*(1.0/self.Rho[1:,:] + 1.0/self.Rho[:-1,:])
        self.Bz[:,:-1] = 0.5*(1.0/self.Rho[:,1:] + 1.0/self.Rho[:,:-1])

    def set_wavelet(self):

        aux = self.set_generical_wavelet()        

        self.wavelet = np.zeros_like(aux)

        for i in range(self.nt):
            self.wavelet[i] = np.sum(aux[:i])        

    def set_wavefield(self):
                
        self.Vx = np.zeros_like(self.Vp)        
        self.Vz = np.zeros_like(self.Vp)        

    def plot_models(self):

        imgs = [self.vp, self.rho]
        pmin = [np.min(self.vp), np.min(self.rho)]
        pmax = [np.max(self.vp), np.max(self.rho)]
        legends = ["P wave velocity [m/s]", "Density [kg/m³]"]

        self.set_generical_model_plot(imgs, legends, pmin, pmax)

    def apply_wavelet(self):
        
        self.P[self.sIdz,self.sIdx] += self.wavelet[self.time_id] / (self.dx*self.dz)

    def solve_8E2T_FDM(self):
        
        dP_dx, dP_dz = get_pressure_derivatives(self.P, self.nxx, self.nzz, self.dx, self.dz)

        self.Vx -= self.dt*self.Bx*dP_dx
        self.Vz -= self.dt*self.Bz*dP_dz

        dVx_dx, dVz_dz = get_velocity_derivatives(self.Vx, self.Vz, self.nxx, self.nzz, self.dx, self.dz)

        self.P -= self.dt*self.K*(dVx_dx + dVz_dz)

    def apply_boundary_condition(self):
        
        self.P *= self.damper
        self.Vx *= self.damper
        self.Vz *= self.damper

@njit(parallel = True)
def get_pressure_derivatives(P, nxx, nzz, dx, dz):

    dP_dx = np.zeros_like(P)
    dP_dz = np.zeros_like(P)

    for index in prange(nxx*nzz):        
        
        i = int(index % nzz)
        j = int(index / nzz)

        if (j >= 4) and (j < nxx - 3):

            dP_dx[i,j] = (75.0*(P[i,j-4] - P[i,j+3]) +
                        1029.0*(P[i,j+2] - P[i,j-3]) +
                        8575.0*(P[i,j-2] - P[i,j+1]) +
                      128625.0*(P[i,j]   - P[i,j-1])) / (dx*107520.0)

        if (i >= 4) and (i < nzz - 3):

            dP_dz[i,j] = (75.0*(P[i-4,j] - P[i+3,j]) + 
                        1029.0*(P[i+2,j] - P[i-3,j]) +
                        8575.0*(P[i-2,j] - P[i+1,j]) +
                      128625.0*(P[i,j]   - P[i-1,j])) / (dz*107520.0)                
    
    return dP_dx, dP_dz

@njit(parallel = True)
def get_velocity_derivatives(Vx, Vz, nxx, nzz, dx, dz):
    
    dVx_dx = np.zeros_like(Vx)
    dVz_dz = np.zeros_like(Vz)

    for index in prange(nxx*nzz):        
        
        i = int(index % nzz)
        j = int(index / nzz)

        if (i >= 3) and (i < nzz - 4) and (j >= 3) and (j < nxx - 4):

            dVx_dx[i,j] = (75.0*(Vx[i,j-3] - Vx[i,j+4]) +   
                         1029.0*(Vx[i,j+3] - Vx[i,j-2]) +
                         8575.0*(Vx[i,j-1] - Vx[i,j+2]) +
                       128625.0*(Vx[i,j+1] - Vx[i,j])) / (dx*107520.0) 

            dVz_dz[i,j] = (75.0*(Vz[i-3,j] - Vz[i+4,j]) + 
                         1029.0*(Vz[i+3,j] - Vz[i-2,j]) +
                         8575.0*(Vz[i-1,j] - Vz[i+2,j]) + 
                       128625.0*(Vz[i+1,j] - Vz[i,j])) / (dz*107520.0)

    return dVx_dx, dVz_dz

