import numpy as np
import matplotlib.pyplot as plt

from numba import njit, prange  

class Wavefield_1D():

    wave_type = "1D wave propagation in constant density acoustic isotropic media"

    def __init__(self):
        
        # TODO: read parameters from a file

        self.nt = 10001
        self.dt = 1e-3
        self.fmax = 30.0

        self.nz = 1001
        self.dz = 5.0

        self.interfaces = np.array([1000, 2000, 3000, 4000])
        self.velocities = np.array([1500, 2000, 2500, 3000, 3500])

        self.depth = np.arange(self.nz) * self.dz
        self.times = np.arange(self.nt) * self.dt

        self.model = self.velocities[0] * np.ones(self.nz)

        self.z_src = np.array([100, 200, 300])
        self.z_rec = np.array([2500, 3500, 4500])

    def get_type(self):
        
        print(self.wave_type)

    def set_model(self):
        
        self.get_type()
        for layerId, interface in enumerate(self.interfaces):
            self.model[int(interface/self.dz):] = self.velocities[layerId+1]    

    def set_wavelet(self):
    
        t0 = 2.0*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0

        arg = np.pi*(np.pi*fc*td)**2.0

        self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)

    def wave_propagation(self):

        self.P = np.zeros((self.nz, self.nt)) # P_{i,n}

        sId = int(self.z_src[0] / self.dz)

        for n in range(1,self.nt-1):

            self.P[sId,n] += self.wavelet[n]    

            laplacian = get_laplacian_1D(self.P, self.dz, self.nz, n)

            self.P[:,n+1] = (self.dt*self.model)**2 * laplacian + 2.0*self.P[:,n] - self.P[:,n-1] 

    def plot_wavefield(self):
        fig, ax = plt.subplots(num = "Wavefield plot", figsize = (8, 8), clear = True)

        ax.imshow(self.P, aspect = "auto", cmap = "Greys")

        # ax.plot(self.P[:,5000])

        ax.set_title("Wavefield", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15) 
        
        fig.tight_layout()
        plt.show()
        
    def plot_model(self):
        
        fig, ax = plt.subplots(num = "Model plot", figsize = (4, 8), clear = True)

        src_projection = np.array(self.z_src / self.dz, dtype = int)
        rec_projection = np.array(self.z_rec / self.dz, dtype = int)

        ax.plot(self.model, self.depth)
        ax.plot(self.model[src_projection], self.z_src, "*", color = "black", label = "Sources")
        ax.plot(self.model[rec_projection], self.z_rec, "v", color = "green", label = "Receivers")
        
        ax.set_title("Model", fontsize = 18)
        ax.set_xlabel("Velocity [m/s]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15) 
        
        ax.set_ylim([0, (self.nz-1)*self.dz])

        ax.legend(loc = "upper right", fontsize = 12)
        
        ax.invert_yaxis()
        fig.tight_layout()
        plt.show()

    def plot_wavelet(self):
       
        fig, ax = plt.subplots(num = "Wavelet plot", figsize = (10, 5), clear = True)

        ax.plot(self.times, self.wavelet)
        ax.set_title("Wavelet", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Amplitude", fontsize = 15) 
        
        ax.set_xlim([0, (self.nt-1)*self.dt])
        
        fig.tight_layout()
        plt.show()


@njit 
def get_laplacian_1D(P, dz, nz, time_id):

    d2P_dz2 = np.zeros(nz)

    for i in prange(1, nz-1): 
        d2P_dz2[i] = (P[i-1,time_id] - 2.0*P[i,time_id] + P[i+1,time_id]) / dz**2.0    

    return d2P_dz2

class Wavefield_2D(Wavefield_1D):
    
    wave_type = "2D wave propagation in constant density acoustic isotropic media"    
    
    def __init__(self):
        
        super().__init__()
        

class Wavefield_3D(Wavefield_2D):

    wave_type = "3D wave propagation in constant density acoustic isotropic media"    

    def __init__(self):
        
        super().__init__()
        
