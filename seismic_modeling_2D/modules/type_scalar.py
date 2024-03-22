import numpy as np

from numba import njit, prange  

from modules.base_modeling import BaseModeling

class Scalar(BaseModeling):

    def __init__(self):
        super().__init__()

        self.name = "Scalar"

    def set_models(self):
        
        self.vp_filename = self.catch_parameter("vp_filename")[0]  
        self.vp, self.Vp = self.set_generical_model(self.vp_filename)    

        self.vp = 1500.0*np.ones((self.nz, self.nx))
        self.Vp = 1500.0*np.ones((self.nzz, self.nxx))

    def set_wavelet(self):

        self.wavelet = self.set_generical_wavelet()        

    def set_wavefield(self):

        self.U_old = np.zeros_like(self.P)
        self.U_fut = np.zeros_like(self.P)

    def plot_models(self):
        
        imgs = [self.vp]
        pmin = [np.min(self.vp)]
        pmax = [np.max(self.vp)]
        legends = ["P wave velocity [m/s]"]

        self.set_generical_model_plot(imgs, legends, pmin, pmax)

    def apply_wavelet(self):

        self.P[self.sIdz,self.sIdx] += self.wavelet[self.time_id] / (self.dx*self.dz)

    def solve_8E2T_FDM(self):
        
        laplacian = get_laplacian_2D(self.P, self.nxx, self.nzz, self.dx, self.dz)

        self.U_fut = laplacian*(self.dt*self.Vp)**2.0 + 2.0*self.P - self.U_old 

        self.U_old = self.P.copy()
        self.P = self.U_fut.copy()

    def apply_boundary_condition(self):
        
        self.P *= self.damper
        self.U_old *= self.damper
        self.U_fut *= self.damper

@njit(parallel = True)
def get_laplacian_2D(P, nxx, nzz, dx, dz):
    
    laplacian = np.zeros_like(P)

    for index in prange(nxx*nzz):        
        
        i = int(index % nzz)
        j = int(index / nzz)

        if (i >= 4) and (i < nzz-4) and (j >= 4) and (j < nxx-4):

            d2P_dx2 = (-  9.0*(P[i,j-4] + P[i,j+4])
                    +   128.0*(P[i,j-3] + P[i,j+3])
                    -  1008.0*(P[i,j-2] + P[i,j+2])
                    +  8064.0*(P[i,j-1] + P[i,j+1])
                    - 14350.0*(P[i,j])) / (5040.0*dx**2.0)

            d2P_dz2 = (-  9.0*(P[i-4,j] + P[i+4,j])
                    +   128.0*(P[i-3,j] + P[i+3,j])
                    -  1008.0*(P[i-2,j] + P[i+2,j])
                    +  8064.0*(P[i-1,j] + P[i+1,j])
                    - 14350.0*(P[i,j])) / (5040.0*dz**2.0);            

            laplacian[i,j] = d2P_dx2 + d2P_dz2

    return laplacian
