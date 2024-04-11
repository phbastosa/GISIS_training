import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from os import system
from numba import njit, prange

from matplotlib import rc
from IPython.display import HTML

rc('animation', html='html5')

class Modeling:

    def __init__(self, parameters):

        self.snapCount = 0    
        self.fdm_stencil = 4    
        self.file = parameters    
    
    def catch_parameter(self, target):
        file = open(self.file, "r")
        for line in file.readlines():
            if line[0] != "#":
                splitted = line.split()
                if len(splitted) != 0:
                    if splitted[0] == target: 
                        return splitted[2]         

    def set_parameters(self):

        self.nx = int(self.catch_parameter("n_samples_x"))
        self.nz = int(self.catch_parameter("n_samples_z"))
        self.nt = int(self.catch_parameter("time_samples"))

        self.dh = float(self.catch_parameter("model_spacing"))
        self.dt = float(self.catch_parameter("time_spacing"))

        self.fmax = float(self.catch_parameter("max_frequency"))

        self.nb = int(self.catch_parameter("boundary_samples"))
        self.factor = float(self.catch_parameter("attenuation_factor")) 

        self.nsnaps = int(self.catch_parameter("total_snapshots"))

        self.vp_file = self.catch_parameter("velocity_model_file")

        self.total_shots = int(self.catch_parameter("total_shots"))
        self.total_nodes = int(self.catch_parameter("total_nodes"))

        self.nxx = self.nx + 2*self.nb
        self.nzz = self.nz + self.nb+4

        self.wavelet = np.zeros(self.nt)
        
        self.vp = np.zeros((self.nz, self.nx))
        self.Vp = np.zeros((self.nzz, self.nxx))

        self.Upas = np.zeros_like(self.Vp)
        self.Upre = np.zeros_like(self.Vp)
        self.Ufut = np.zeros_like(self.Vp)

        self.damp2D = np.ones_like(self.Vp)

        self.snapshots = np.zeros((self.nz, self.nx, self.nsnaps))
        self.seismogram = np.zeros((self.nt, self.total_nodes))

    def set_geometry(self):

        ishot = float(self.catch_parameter("shot_beg"))
        fshot = float(self.catch_parameter("shot_end"))
        selev = float(self.catch_parameter("shot_elevation"))

        self.sx = np.linspace(ishot, fshot, self.total_shots)
        self.sz = np.ones(self.total_shots) * selev 

        inode = float(self.catch_parameter("node_beg"))
        fnode = float(self.catch_parameter("node_end"))
        gelev = float(self.catch_parameter("node_elevation"))

        self.rx = np.linspace(inode, fnode, self.total_nodes)
        self.rz = np.ones(self.total_nodes) * gelev 

    def import_model(self):
        data = np.fromfile(self.vp_file, dtype = np.float32, count = self.nx * self.nz)            
        self.vp = np.reshape(data, [self.nz, self.nx], order = "F")

    def set_wavelet(self):

        fc = self.fmax / (3.0 * np.sqrt(np.pi))    

        self.tlag = 2.0*np.pi/self.fmax

        for n in range(self.nt):
            aux = np.pi*((n*self.dt - self.tlag)*fc*np.pi) ** 2.0 
            self.wavelet[n] = (1.0 - 2.0*aux)*np.exp(-aux)

        w = 2.0*np.pi*np.fft.fftfreq(self.nt, self.dt)

        self.wavelet = np.real(np.fft.ifft(np.fft.fft(self.wavelet)*np.sqrt(complex(0,1)*w))) 

    def set_boundary(self):

        self.Vp[self.fdm_stencil:self.nzz-self.nb, self.nb:self.nxx-self.nb] = self.vp[:,:]

        self.Vp[:self.fdm_stencil, self.nb:self.nxx-self.nb] = self.vp[0,:]

        for i in range(self.nb):
            self.Vp[self.nzz-i-1, self.nb:self.nxx-self.nb] = self.vp[-1,:]

        for j in range(self.nb):
            self.Vp[:,j] = self.Vp[:,self.nb]
            self.Vp[:,self.nxx-j-1] = self.Vp[:,-(self.nb+1)]

        self.vmax = np.max(self.Vp)
        self.vmin = np.min(self.Vp)

    def set_damper(self):
        
        damp1D = np.zeros(self.nb)

        for i in range(self.nb):   
            damp1D[i] = np.exp(-(self.factor*(self.nb - i))**2.0)

        for i in range(0, self.nzz-self.nb):
            self.damp2D[i, :self.nb] = damp1D
            self.damp2D[i, self.nxx-self.nb:self.nxx] = damp1D[::-1]

        for j in range(self.nb, self.nxx-self.nb):
            self.damp2D[self.nzz-self.nb:self.nzz, j] = damp1D[::-1]    

        for i in range(self.nb):
            self.damp2D[self.nzz-self.nb-1:self.nzz-i-1, i] = damp1D[i]
            self.damp2D[self.nzz-i-1, i:self.nb] = damp1D[i]

            self.damp2D[self.nzz-self.nb-1:self.nzz-i, self.nxx-i-1] = damp1D[i]
            self.damp2D[self.nzz-i-1, self.nxx-self.nb-1:self.nxx-i] = damp1D[i]

    def fdm_propagation(self):

        sIdx = int(self.sx[0] / self.dh) + self.nb
        sIdz = int(self.sz[0] / self.dh + self.fdm_stencil)    

        for self.time_step in range(self.nt):

            self.show_modeling_status()

            self.Upre[sIdz,sIdx] += self.wavelet[self.time_step] / (self.dh*self.dh)

            laplacian = fdm_8E2T_scalar2D(self.Upre, self.nxx, self.nzz, self.dh)
            
            self.Ufut = laplacian*(self.dt*self.dt*self.Vp*self.Vp) - self.Upas + 2.0*self.Upre

            self.Upas = self.Upre * self.damp2D     
            self.Upre = self.Ufut * self.damp2D

            self.get_snapshots()
            self.get_seismogram()

    def show_modeling_status(self):

        beta = 2
        alpha = 3

        if self.time_step % int(self.nt / 100) == 0:
            system("clear")
            print(f"\nModel lenght = ({(self.nz-1)*self.dh:.0f}, {(self.nx-1)*self.dh:.0f}) m")
            print(f"Spacial spacing = {self.dh} m")
            print(f"Max velocity = {self.vmax} m/s")
            print(f"Min velocity = {self.vmin} m/s")
            
            print(f"\nTime step = {self.dt} s")
            print(f"Total time = {(self.nt-1)*self.dt} s")
            print(f"Max frequency = {self.fmax} Hz")

            print(f"\nHighest frequency without dispersion:")
            print(f"Frequency <= {self.vmin / (alpha*self.dh):.1f} Hz")

            print(f"\nHighest time step without instability:")
            print(f"Time step <= {self.dh / (beta*self.vmax):.5f} s")

            print(f"\nModeling progress = {100*(self.time_step+1)/self.nt:.0f} %")        

    def get_snapshots(self):        
        if self.snapCount < self.nsnaps:
            if self.time_step % int(self.nt / self.nsnaps) == 0:
                self.snapshots[:, :, self.snapCount] = self.Upre[self.fdm_stencil:self.nz+self.fdm_stencil, self.nb:self.nxx-self.nb]
                self.snapCount += 1

    def get_seismogram(self):
        for k in range(self.total_nodes):
            rIdx = int(self.rx[k] / self.dh) + self.nb
            rIdz = int(self.rz[k] / self.dh) + 4
            self.seismogram[self.time_step, k] = self.Upre[rIdz, rIdx]
    
    def plot_wavelet(self):
        nw = int(4*self.tlag/self.dt)
        t = np.arange(nw) * self.dt
        f = np.fft.fftfreq(self.nt, self.dt)

        fwavelet = np.fft.fft(self.wavelet)

        fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10,8))

        ax[0].plot(t, self.wavelet[:nw] * 1.0 / np.max(self.wavelet))
        ax[0].set_xlim([0, nw*self.dt])
        ax[0].set_title("Wavelet filtered by half derivative technique", fontsize = 18)    
        ax[0].set_xlabel("Time [s]", fontsize = 15)    
        ax[0].set_ylabel("Normalized amplitude", fontsize = 18)    

        ax[1].plot(f, np.abs(fwavelet) * 1.0 / np.max(np.abs(fwavelet)), "o")
        ax[1].set_xlim([0, self.fmax])
        ax[1].set_title("Wavelet in frequency domain", fontsize = 18)    
        ax[1].set_xlabel("Frequency [Hz]", fontsize = 15)    
        ax[1].set_ylabel("Normalized amplitude", fontsize = 18)    

        plt.tight_layout()
        plt.show()

    def plot_geometry(self):

        xloc = np.linspace(self.nb, self.nx + self.nb - 1, 11, dtype = int) 
        xlab = np.array((xloc - self.nb)*self.dh, dtype = int) 

        zloc = np.linspace(0, self.nz - 1, 7, dtype = int) 
        zlab = np.array(zloc*self.dh, dtype = int) 

        fig, ax = plt.subplots(1,1, figsize = (12,5))

        img = ax.imshow(self.Vp, aspect = "auto", cmap = "jet")
        cbar = fig.colorbar(img, ax = ax, extend = 'neither')
        cbar.minorticks_on()

        ax.plot(np.arange(self.nx)+self.nb, np.ones(self.nx)+4-1, "-k")
        ax.plot(np.arange(self.nx)+self.nb, np.ones(self.nx)+4+self.nz-1, "-k")
        ax.plot(np.ones(self.nz)+self.nb-1, np.arange(self.nz)+4, "-k")
        ax.plot(np.ones(self.nz)+self.nb+self.nx-1, np.arange(self.nz)+4, "-k")

        ax.scatter(self.sx / self.dh + self.nb, self.sz / self.dh + self.fdm_stencil, color = "red")
        ax.scatter(self.rx / self.dh + self.nb, self.rz / self.dh + self.fdm_stencil, color = "cyan")
        
        ax.set_xticks(xloc)
        ax.set_xticklabels(xlab)

        ax.set_yticks(zloc)
        ax.set_yticklabels(zlab)

        ax.set_title("Marmousi2 model, delimitations and geometry", fontsize = 18)
        ax.set_xlabel("Distance [m]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15)

        plt.tight_layout()
        plt.show()

    def plot_seismogram(self):

        tloc = np.linspace(0, self.nt, 11, dtype = int)
        tlab = np.around(tloc * self.dt, decimals = 1)

        xloc = np.linspace(0, self.total_nodes-1, 9)
        xlab = np.array((self.rx[1] - self.rx[0])*xloc + self.rx[0], dtype = int)

        scale = 0.9 * np.std(self.seismogram)    

        fig, ax = plt.subplots(figsize = (10,8))

        img = ax.imshow(self.seismogram, aspect = "auto", cmap="Greys", vmin = -scale, vmax = scale)
        cbar = fig.colorbar(img, ax = ax, extend = 'neither')
        cbar.minorticks_on()

        ax.set_yticks(tloc)
        ax.set_yticklabels(tlab)

        ax.set_xticks(xloc)
        ax.set_xticklabels(xlab)

        ax.set_title("Seismogram", fontsize = 18)
        ax.set_xlabel("Distance [m]", fontsize = 15)
        ax.set_ylabel("Two Way Time [s]", fontsize = 15)

        plt.tight_layout()
        plt.show()

    def plot_snapshots(self):

        self.scale = 0.5 * np.std(self.snapshots)

        xloc = np.linspace(0, self.nx-1, 11, dtype = int) 
        xlab = np.array(xloc*self.dh, dtype = int) 

        zloc = np.linspace(0, self.nz - 1, 7, dtype = int) 
        zlab = np.array(zloc*self.dh, dtype = int) 

        fig, self.ax = plt.subplots(figsize = (12,5))

        self.shots = self.ax.scatter(self.sx / self.dh, self.sz / self.dh, color = "red")
        self.nodes = self.ax.scatter(self.rx / self.dh, self.rz / self.dh, color = "cyan")

        self.vmod = self.ax.imshow(self.vp, aspect = "auto", cmap = "jet", alpha = 0.3) 

        ims = []
        for i in range(self.nsnaps):
            self.snap = self.ax.imshow(self.snapshots[:,:,i], aspect = "auto", cmap = "Greys", vmin = -self.scale, vmax = self.scale)
            im = [self.snap, self.vmod, self.shots, self.nodes]
            ims.append(im)

        self.ani = animation.ArtistAnimation(fig, ims, interval = 100, blit=True, repeat_delay = 0)
 
        self.ax.set_xticks(xloc)
        self.ax.set_xticklabels(xlab)
        self.ax.set_xlabel("Distance [m]", fontsize = 15)

        self.ax.set_yticks(zloc)
        self.ax.set_yticklabels(zlab)
        self.ax.set_ylabel("Depth [m]", fontsize = 15)
        fig.tight_layout()
        plt.show()
        

@njit(parallel = True)
def fdm_8E2T_scalar2D(Upre, nxx, nzz, dh):

    laplacian = np.zeros_like(Upre)

    for index in prange(nxx*nzz):
        
        i = int(index % nzz)
        j = int(index / nzz)

        if (3 < i < nzz - 4) and (3 < j < nxx - 4):
            
            d2U_dx2 = (- 9.0*(Upre[i, j - 4] + Upre[i, j + 4]) \
                   +   128.0*(Upre[i, j - 3] + Upre[i, j + 3]) \
                   -  1008.0*(Upre[i, j - 2] + Upre[i, j + 2]) \
                   +  8064.0*(Upre[i, j + 1] + Upre[i, j - 1]) \
                   - 14350.0*(Upre[i, j])) / (5040.0*dh*dh)

            d2U_dz2 = (- 9.0*(Upre[i - 4, j] + Upre[i + 4, j]) \
                   +   128.0*(Upre[i - 3, j] + Upre[i + 3, j]) \
                   -  1008.0*(Upre[i - 2, j] + Upre[i + 2, j]) \
                   +  8064.0*(Upre[i - 1, j] + Upre[i + 1, j]) \
                   - 14350.0*(Upre[i, j])) / (5040.0*dh*dh)

            laplacian[i,j] = d2U_dx2 + d2U_dz2

    return laplacian