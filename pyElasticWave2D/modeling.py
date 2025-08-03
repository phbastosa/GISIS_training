import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from os import system
from numba import njit, prange

from matplotlib import rc

rc('animation', html = 'html5')

class Modeling:

    def __init__(self, parameters):

        self.snapCount = 0    
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
        
        self.vp_file = self.catch_parameter("vp_file")
        self.vs_file = self.catch_parameter("vs_file")
        self.ro_file = self.catch_parameter("ro_file")

        self.total_shots = int(self.catch_parameter("total_shots"))
        self.total_nodes = int(self.catch_parameter("total_nodes"))

        self.nxx = self.nx + 2*self.nb
        self.nzz = self.nz + 2*self.nb

        self.wavelet = np.zeros(self.nt)
        
        self.vp = np.zeros((self.nz, self.nx))
        self.vs = np.zeros((self.nz, self.nx))
        self.ro = np.zeros((self.nz, self.nx))
        
        self.B = np.zeros((self.nzz, self.nxx))
        self.M = np.zeros((self.nzz, self.nxx))
        self.L = np.zeros((self.nzz, self.nxx))

        self.P = np.zeros_like(self.B)
        self.Vx = np.zeros_like(self.B)
        self.Vz = np.zeros_like(self.B)
        self.Txx = np.zeros_like(self.B)
        self.Tzz = np.zeros_like(self.B)
        self.Txz = np.zeros_like(self.B)

        self.damp2D = np.ones_like(self.B)

        self.snapshots = np.zeros((self.nz, self.nx, self.nsnaps))
        self.seismogram = np.zeros((self.nt, self.total_nodes))

    def import_models(self):

        self.vp = np.fromfile(self.vp_file, dtype = np.float32, count = self.nx * self.nz).reshape([self.nz, self.nx], order = "F")
        self.vs = np.fromfile(self.vs_file, dtype = np.float32, count = self.nx * self.nz).reshape([self.nz, self.nx], order = "F")
        self.ro = np.fromfile(self.ro_file, dtype = np.float32, count = self.nx * self.nz).reshape([self.nz, self.nx], order = "F")

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

    def set_boundary(self):

        x_slice = slice(self.nb,self.nxx-self.nb)
        z_slice = slice(self.nb,self.nzz-self.nb)

        self.B[z_slice, x_slice] = 1.0 / self.ro
        self.M[z_slice, x_slice] = self.ro*self.vs**2
        self.L[z_slice, x_slice] = self.ro*(self.vp**2 - self.vs**2)

        for i in range(self.nb):

            self.B[i, x_slice] = self.B[self.nb, x_slice]
            self.B[self.nzz-i-1, x_slice] = self.B[self.nzz-self.nb-1, x_slice]

            self.M[i, x_slice] = self.M[self.nb, x_slice]
            self.M[self.nzz-i-1, x_slice] = self.M[self.nzz-self.nb-1, x_slice]

            self.L[i, x_slice] = self.L[self.nb, x_slice]
            self.L[self.nzz-i-1, x_slice] = self.L[self.nzz-self.nb-1, x_slice]

        for j in range(self.nb):

            self.B[:,j] = self.B[:,self.nb]
            self.B[:,self.nxx-j-1] = self.B[:,-(self.nb+1)]

            self.M[:,j] = self.M[:,self.nb]
            self.M[:,self.nxx-j-1] = self.M[:,-(self.nb+1)]

            self.L[:,j] = self.L[:,self.nb]
            self.L[:,self.nxx-j-1] = self.L[:,-(self.nb+1)]

        self.vmax = np.max(self.vp)
        self.vmin = np.min(self.vp)

    def set_wavelet(self):

        fc = self.fmax / (3.0 * np.sqrt(np.pi))    

        self.tlag = 2.0*np.pi/self.fmax

        arg = np.pi*((np.arange(self.nt)*self.dt - self.tlag)*fc*np.pi) ** 2.0 
        aux = (1.0 - 2.0*arg)*np.exp(-arg)

        w = 2.0*np.pi*np.fft.fftfreq(self.nt, self.dt)

        aux = np.real(np.fft.ifft(np.fft.fft(aux)*np.sqrt(complex(0,1)*w))) 

        for n in range(self.nt):
            self.wavelet[n] = np.sum(aux[:n+1])

    def set_damper(self):
        
        damp1D = np.zeros(self.nb)

        for i in range(self.nb):   
            damp1D[i] = np.exp(-(self.factor*(self.nb - i))**2.0)

        for i in range(self.nzz):
            self.damp2D[i,:self.nb] *= damp1D
            self.damp2D[i,-self.nb:] *= damp1D[::-1]

        for j in range(self.nxx):
            self.damp2D[:self.nb,j] *= damp1D
            self.damp2D[-self.nb:,j] *= damp1D[::-1]    

    def fdm_propagation(self):

        for self.sId in range(self.total_shots):

            sIdx = int(self.sx[self.sId] / self.dh) + self.nb
            sIdz = int(self.sz[self.sId] / self.dh) + self.nb    

            self.Vx = np.zeros((self.nzz,self.nxx))
            self.Vz = np.zeros((self.nzz,self.nxx))
            self.Txx = np.zeros((self.nzz,self.nxx))
            self.Tzz = np.zeros((self.nzz,self.nxx))
            self.Txz = np.zeros((self.nzz,self.nxx))

            for self.time_step in range(self.nt):

                self.show_modeling_status()

                self.Txx[sIdz,sIdx] += self.wavelet[self.time_step] / (self.dh*self.dh)
                self.Tzz[sIdz,sIdx] += self.wavelet[self.time_step] / (self.dh*self.dh)

                fdm2D_8E2T_velocity(self.Vx, self.Vz, self.Txx, self.Tzz, self.Txz, self.B, self.P, self.nxx, self.nzz, self.dt, self.dh)
                fdm2D_8E2T_pressure(self.Vx, self.Vz, self.Txx, self.Tzz, self.Txz, self.M, self.L, self.nxx, self.nzz, self.dt, self.dh)

                self.Vx *= self.damp2D
                self.Vz *= self.damp2D
                self.Txx *= self.damp2D
                self.Tzz *= self.damp2D
                self.Txz *= self.damp2D

                self.get_snapshots()
                self.get_seismogram()

    def show_modeling_status(self):

        if self.time_step % int(self.nt / 100) == 0:
            system("clear")
            print(f"\nModel size = ({(self.nz-1)*self.dh:.0f}, {(self.nx-1)*self.dh:.0f}) m")
            print(f"Model spacing = {self.dh} m")
            print(f"Max velocity = {self.vmax} m/s")
            print(f"Min velocity = {self.vmin} m/s")
            
            print(f"\nTime step = {self.dt} s")
            print(f"Total time = {(self.nt-1)*self.dt} s")
            print(f"Max frequency = {self.fmax} Hz")

            print(f"\nRunning shot {self.sId+1} of {self.total_shots}")

            print(f"\nModeling progress = {100*(self.time_step+1)/self.nt:.0f} %")        

    def get_snapshots(self):        
        if self.snapCount < self.nsnaps:
            if self.time_step % int(self.nt / self.nsnaps) == 0:
                self.snapshots[:, :, self.snapCount] = self.P[self.nb:-self.nb, self.nb:-self.nb]
                self.snapCount += 1

    def get_seismogram(self):
        for k in range(self.total_nodes):
            rIdx = int(self.rx[k] / self.dh) + self.nb
            rIdz = int(self.rz[k] / self.dh) + self.nb
            self.seismogram[self.time_step, k] = self.P[rIdz, rIdx]
    
    def plot_wavelet(self):

        t = np.arange(self.nt) * self.dt
        f = np.fft.fftfreq(self.nt, self.dt)

        fwavelet = np.fft.fft(self.wavelet)

        fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10,8))

        ax[0].plot(t, self.wavelet * 1.0 / np.max(self.wavelet))
        ax[0].set_xlim([0, (self.nt-1)*self.dt])
        ax[0].set_title("Wavelet filtered by half derivative technique", fontsize = 18)    
        ax[0].set_xlabel("Time [s]", fontsize = 15)    
        ax[0].set_ylabel("Normalized amplitude", fontsize = 18)    

        ax[1].plot(f, np.abs(fwavelet) * 1.0 / np.max(np.abs(fwavelet)), "o")
        ax[1].set_xlim([0, self.fmax])
        ax[1].set_title("Wavelet in frequency domain", fontsize = 18)    
        ax[1].set_xlabel("Frequency [Hz]", fontsize = 15)    
        ax[1].set_ylabel("Normalized amplitude", fontsize = 18)    

        fig.tight_layout()
        plt.show()

    def plot_seismogram(self):

        tloc = np.linspace(0, self.nt-1, 11, dtype = int)
        tlab = np.around(tloc * self.dt, decimals = 1)

        xloc = np.linspace(0, self.total_nodes-1, 9)
        xlab = np.array((self.rx[1] - self.rx[0])*xloc + self.rx[0], dtype = int)

        scale = 2.0 * np.std(self.seismogram)    

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

        self.scale = 2.0 * np.std(self.snapshots)

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

        self.ani = animation.ArtistAnimation(fig, ims, interval = (self.nt/self.nsnaps)*self.dt*1e3, blit = True, repeat_delay = 0)
 
        self.ax.set_xticks(xloc)
        self.ax.set_xticklabels(xlab)
        self.ax.set_xlabel("Distance [m]", fontsize = 15)

        self.ax.set_yticks(zloc)
        self.ax.set_yticklabels(zlab)
        self.ax.set_ylabel("Depth [m]", fontsize = 15)
        fig.tight_layout()
        plt.show()
        

@njit(parallel = True)
def fdm2D_8E2T_velocity(Vx, Vz, Txx, Tzz, Txz, B, P, nxx, nzz, dt, dh):

    for index in prange(nxx*nzz):
        
        i = int(index % nzz)
        j = int(index / nzz)

        if (3 <= i < nzz - 4) and (3 < j < nxx - 3):
            
            d_Txx_dx = (75.0*(Txx[i, j - 4] - Txx[i, j + 3]) +
                      1029.0*(Txx[i, j + 2] - Txx[i, j - 3]) +
                      8575.0*(Txx[i, j - 2] - Txx[i, j + 1]) +
                    128625.0*(Txx[i, j]     - Txx[i, j - 1])) / (107520.0*dh)

            d_Txz_dz = (75.0*(Txz[i - 3, j] - Txz[i + 4, j]) +
                      1029.0*(Txz[i + 3, j] - Txz[i - 2, j]) +
                      8575.0*(Txz[i - 1, j] - Txz[i + 2, j]) +
                    128625.0*(Txz[i + 1, j] - Txz[i, j])) / (107520.0*dh)

            Bx = 0.5*(B[i,j+1] + B[i,j])

            Vx[i,j] += dt*Bx*(d_Txx_dx + d_Txz_dz) 

        if (3 < i < nzz - 3) and (3 <= j < nxx - 4):

            d_Txz_dx = (75.0*(Txz[i, j - 3] - Txz[i, j + 4]) +
                      1029.0*(Txz[i, j + 3] - Txz[i, j - 2]) +
                      8575.0*(Txz[i, j - 1] - Txz[i, j + 2]) +
                    128625.0*(Txz[i, j + 1] - Txz[i, j])) / (107520.0*dh)

            d_Tzz_dz = (75.0*(Tzz[i - 4, j] - Tzz[i + 3, j]) +
                      1029.0*(Tzz[i + 2, j] - Tzz[i - 3, j]) +
                      8575.0*(Tzz[i - 2, j] - Tzz[i + 1, j]) +
                    128625.0*(Tzz[i, j]     - Tzz[i - 1, j])) / (107520.0*dh)

            Bz = 0.5*(B[i+1,j] + B[i,j])

            Vz[i,j] += dt*Bz*(d_Txz_dx + d_Tzz_dz)

        P[i,j] = 0.5*(Txx[i,j] + Tzz[i,j])

@njit(parallel = True)
def fdm2D_8E2T_pressure(Vx, Vz, Txx, Tzz, Txz, M, L, nxx, nzz, dt, dh):

    for index in prange(nxx*nzz):
        
        i = int(index % nzz)
        j = int(index / nzz)

        if (3 <= i < nzz - 4) and (3 <= j < nxx - 4):
            
            d_Vx_dx = (75.0*(Vx[i, j - 3] - Vx[i, j + 4]) +
                     1029.0*(Vx[i, j + 3] - Vx[i, j - 2]) +
                     8575.0*(Vx[i, j - 1] - Vx[i, j + 2]) +
                   128625.0*(Vx[i, j + 1] - Vx[i, j])) / (107520.0*dh)

            d_Vz_dz = (75.0*(Vz[i - 3, j] - Vz[i + 4, j]) +
                     1029.0*(Vz[i + 3, j] - Vz[i - 2, j]) +
                     8575.0*(Vz[i - 1, j] - Vz[i + 2, j]) +
                   128625.0*(Vz[i + 1, j] - Vz[i, j])) / (107520.0*dh)

            Txx[i,j] += dt*((L[i,j] + 2*M[i,j])*d_Vx_dx + L[i,j]*d_Vz_dz) 
            Tzz[i,j] += dt*((L[i,j] + 2*M[i,j])*d_Vz_dz + L[i,j]*d_Vx_dx) 

        if (3 < i < nzz - 3) and (3 < j < nxx - 3):

            d_Vz_dx = (75.0*(Vz[i, j - 4] - Vz[i, j + 3]) +
                     1029.0*(Vz[i, j + 2] - Vz[i, j - 3]) +
                     8575.0*(Vz[i, j - 2] - Vz[i, j + 1]) +
                   128625.0*(Vz[i, j]     - Vz[i, j - 1])) / (107520.0*dh)

            d_Vx_dz = (75.0*(Vx[i - 4, j] - Vx[i + 3, j]) +
                     1029.0*(Vx[i + 2, j] - Vx[i - 3, j]) +
                     8575.0*(Vx[i - 2, j] - Vx[i + 1, j]) +
                   128625.0*(Vx[i, j]     - Vx[i - 1, j])) / (107520.0*dh)

            Mxz = pow(0.25*(1.0/M[i+1,j+1] + 1.0/M[i,j+1] + 
                            1.0/M[i+1,j]   + 1.0/M[i,j]), -1.0)

            Txz[i,j] += dt*Mxz*(d_Vx_dz + d_Vz_dx)
