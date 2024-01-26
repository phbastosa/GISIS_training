import numpy as np
import matplotlib.pyplot as plt

class Wavefield_1D():

    wave_type = "1D wave propagation in constant density acoustic isotropic media"

    def __init__(self):
        
        # TODO: read parameters from a file

        self.nt = 1001
        self.dt = 1e-3
        self.fmax = 30.0

        self.nz = 1001
        self.dz = 5.0

        self.interfaces = np.array([1000, 2000, 3000, 4000])
        self.velocities = np.array([1500, 2000, 2500, 3000, 3500])

        self.depth = np.arange(self.nz) * self.dz
        self.times = np.arange(self.nt) * self.dt

        self.model = self.velocities[0] * np.ones(self.nz)

    def get_type(self):
        
        print(self.wave_type)

    def set_model(self):
        
        for layerId, interface in enumerate(self.interfaces):
            self.model[int(interface/self.dz):] = self.velocities[layerId+1]    

    def set_wavelet(self):
    
        t0 = 2.0*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0

        arg = np.pi*(np.pi*fc*td)**2.0

        self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)

    def plot_model(self):
        
        fig, ax = plt.subplots(num = "Model plot", figsize = (4, 8), clear = True)

        ax.plot(self.model, self.depth)
        ax.set_title("Model", fontsize = 18)
        ax.set_xlabel("Velocity [m/s]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15) 
        
        ax.set_ylim([0, (self.nz-1)*self.dz])
        
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


class Wavefield_2D(Wavefield_1D):
    
    wave_type = "2D wave propagation in constant density acoustic isotropic media"    
    
    def __init__(self):
        
        super().__init__()
        

class Wavefield_3D(Wavefield_2D):

    wave_type = "3D wave propagation in constant density acoustic isotropic media"    

    def __init__(self):
        
        super().__init__()
        
