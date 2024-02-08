import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as movie

from numba import njit, prange  

class Wavefield_1D():

    wave_type = "1D wave propagation in constant density acoustic isotropic media"

    def __init__(self):
        
        # TODO: read parameters from a file

        self.nt = 10001
        self.dt = 1e-3
        self.fmax = 10.0

        self.nz = 501
        self.dz = 5.0

        self.interfaces = np.array([])
        self.velocities = np.array([1500])

        self.depth = np.arange(self.nz) * self.dz
        self.times = np.arange(self.nt) * self.dt

        self.vp = self.velocities[0] * np.ones(self.nz)

        self.z_src = np.array([(0.4*self.nz - 1) * self.dz])
        self.z_rec = np.array([(0.8*self.nz - 1) * self.dz])

        self.src_projection = np.array(self.z_src / self.dz, dtype = int)
        self.rec_projection = np.array(self.z_rec / self.dz, dtype = int)

    def set_model(self):
        
        print(self.wave_type)
        for layerId, interface in enumerate(self.interfaces):
            self.vp[int(interface/self.dz):] = self.velocities[layerId+1]    

    def set_wavelet(self):
    
        t0 = 2.0*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0

        arg = np.pi*(np.pi*fc*td)**2.0

        self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)

    def wave_propagation(self):

        self.P = np.zeros((self.nz, self.nt))

        sId = int(self.z_src[0] / self.dz)

        for n in range(1,self.nt-1):

            self.P[sId,n] += self.wavelet[n]    

            laplacian = get_laplacian_1D(self.P, self.nz, self.dz, n)

            self.P[:,n+1] = (self.dt*self.vp)**2 * laplacian + 2.0*self.P[:,n] - self.P[:,n-1] 

        self.P *= 1.0 / np.max(self.P) 

        self.seismogram = self.P[self.rec_projection,:].T 

    def plot_wavefield(self):
        
        zloc = np.linspace(0, self.nz-1, 5)
        zlab = np.array(zloc * self.dz, dtype = int)

        tloc = np.linspace(0, self.nt-1, 11)
        tlab = np.array(tloc * self.dt, dtype = int)    

        fig, ax = plt.subplots(num = "Wavefield plot", figsize = (8, 8), clear = True)

        ax.imshow(self.P, aspect = "auto", cmap = "Greys")

        ax.set_xticks(tloc)
        ax.set_yticks(zloc)
        ax.set_xticklabels(tlab)
        ax.set_yticklabels(zlab)

        ax.set_title("Wavefield", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15) 
        
        fig.tight_layout()
        plt.show()
        
    def plot_model(self):
        
        fig, ax = plt.subplots(num = "Model plot", figsize = (4, 8), clear = True)

        ax.plot(self.vp, self.depth)
        ax.plot(self.vp[self.src_projection], self.z_src, "*", color = "black", label = "Sources")
        ax.plot(self.vp[self.rec_projection], self.z_rec, "v", color = "green", label = "Receivers")
        
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

    def plot_seismogram(self):

        fig, ax = plt.subplots(num = "Seismogram plot", figsize = (4, 8), clear = True)

        ax.plot(self.seismogram, self.times)
        
        ax.set_title("Seismogram", fontsize = 18)
        ax.set_xlabel("Normalized Amplitude", fontsize = 15)
        ax.set_ylabel("Time [s]", fontsize = 15) 
        
        ax.set_ylim([0, (self.nt-1)*self.dt])
        
        ax.invert_yaxis()
        fig.tight_layout()
        plt.show()

    def plot_wave_propagation(self):

        fig, ax = plt.subplots(num = "Wave animation", figsize = (4, 8), clear = True)

        ax.set_title("Wave propagation", fontsize = 18)
        ax.set_ylabel("Depth [m]", fontsize = 15)
        ax.set_xlabel("Amplitude", fontsize = 15) 
        
        ax.set_ylim([0, (self.nz-1)*self.dz])
        ax.set_xlim([np.min(self.P)-0.5, np.max(self.P)+0.5])

        wave, = ax.plot(self.P[:,0], self.depth)
        srcs, = ax.plot(self.P[self.src_projection,0], self.z_src, "*", color = "black", label = "Sources")
        recs, = ax.plot(self.P[self.rec_projection,0], self.z_rec, "v", color = "green", label = "Receivers")

        artists_list = []

        artists_list.append(wave)
        artists_list.append(srcs)
        artists_list.append(recs)

        def init():
            wave.set_ydata(self.depth)  
            wave.set_xdata([np.nan] * self.nz)
                        
            return artists_list

        def animate(i): 
            wave.set_ydata(self.depth)  
            wave.set_xdata(self.P[:,i])
        
            srcs.set_xdata(self.P[self.src_projection,i])
            recs.set_xdata(self.P[self.rec_projection,i]) 

            return artists_list

        ax.legend(loc = "upper right")
        ax.invert_yaxis()
        fig.tight_layout()

        _= movie.FuncAnimation(fig, animate, init_func = init, interval = 1e-3*self.dt, frames = self.nt, blit = True, repeat = True)
        
        plt.show()

@njit 
def get_laplacian_1D(P, nz, dz, n):

    d2P_dz2 = np.zeros(nz)
    for i in prange(1, nz-1): 
        d2P_dz2[i] = (P[i-1,n] - 2.0*P[i,n] + P[i+1,n]) / dz**2.0    

    return d2P_dz2



class Wavefield_2D(Wavefield_1D):
    
    wave_type = "2D wave propagation in constant density acoustic isotropic media"    
    
    def __init__(self):
        
        super().__init__()
        

class Wavefield_3D(Wavefield_2D):

    wave_type = "3D wave propagation in constant density acoustic isotropic media"    

    def __init__(self):
        
        super().__init__()
        
