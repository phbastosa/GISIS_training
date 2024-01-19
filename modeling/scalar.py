import numpy as np
import matplotlib.pyplot as plt

class Wavefield_1D():

    wave_type = "1D wave propagation in constant density acoustic isotropic media"

    def __init__(self):
        
        # TODO: read parameters from a file

        self.nt = 1001
        self.dt = 1e-3
        self.fmax = 30.0

        # self.nz
        # self.dz

        # self.model = np.zeros(nz)
        # self.depth = np.arange(nz) * self.dz

        # self.interfaces = [z1, z2, z3, ..., zn]
        # self.velocities = [v1, v2, v3, ..., vn]

    def set_model(self):
        pass    

    def plot_model(self):
        pass


    def get_type(self):
        print(self.wave_type)

    def set_wavelet(self):
    
        t0 = 2.0*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0

        arg = np.pi*(np.pi*fc*td)**2.0

        self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)

    def plot_wavelet(self):
       
        t = np.arange(self.nt)*self.dt

        fig, ax = plt.subplots(figsize = (10, 5), clear = True)

        ax.plot(t, self.wavelet)
        ax.set_title("Wavelet", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Amplitude", fontsize = 15) 
        
        ax.set_xlim([0, np.max(t)])
        
        fig.tight_layout()
        plt.show()


class Wavefield_2D(Wavefield_1D):
    
    wave_type = "2D wave propagation in constant density acoustic isotropic media"    
    
    def __init__(self):
        super().__init__()
        

class Wavefield_3D(Wavefield_2D):

    wave_type = "3D wave propagation in constant density acoustic isotropic media"    

    def __init__(self):
        super().__init__()
        
