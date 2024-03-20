from os import system
from sys import argv
from time import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

class BaseModeling():

    def __init__(self):    

        self.name = None

        self.parameter_file = argv[1]  

        self.modeling_type = int(self.catch_parameter("modeling_type")[0])

    def catch_parameter(self, target):

        file = open(self.parameter_file, 'r')
        for line in file.readlines():
            if line[0] != '#':
                splitted = line.split()
                if len(splitted) != 0:
                    if splitted[0] == target: 
                        return splitted[2:]

    def import_binary_matrix(self, n1, n2, filename):

        data = np.fromfile(filename, dtype = np.float32, count = n1*n2)        
        return np.reshape(data, [n1, n2], order = "F")

    def set_generical_model(self, filename):
        
        model = self.import_binary_matrix(self.nz, self.nx, filename) 

        Model = np.zeros((self.nzz, self.nxx))

        Model[self.nabc:self.nzz-self.nabc,self.nabc:self.nxx-self.nabc] = model.copy()

        for i in range(self.nabc):
            Model[i,self.nabc:self.nxx-self.nabc] = model[0,:]
            Model[self.nzz-i-1,self.nabc:self.nxx-self.nabc] = model[-1,:]

        for i in range(self.nabc):
            Model[:,i] = Model[:,self.nabc]
            Model[:,self.nxx-i-1] = Model[:,self.nxx-self.nabc-1]
    
        return model, Model

    def set_generical_wavelet(self):

        t0 = 1.5*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0

        arg = np.pi*(np.pi*fc*td)**2.0

        wavelet = (1.0 - 2.0*arg)*np.exp(-arg)
        omega = 2.0*np.pi*np.fft.fftfreq(self.nt, self.dt)

        return np.real(np.fft.ifft(np.fft.fft(wavelet) * np.sqrt(complex(0,1)*omega)))
         
    def set_generical_model_plot(self, imgs, legends, pmin, pmax):
        
        xdim = 10
        zdim = (0.5 + len(imgs)) * self.nz/self.nx * xdim     

        xloc = np.linspace(0, self.nx, 11)
        xlab = np.linspace(0,(self.nx-1)*self.dx, 11, dtype = int)

        zloc = np.linspace(0, self.nz, 5)
        zlab = np.linspace(0,(self.nz-1)*self.dz, 5, dtype = int)

        fig, ax = plt.subplots(num = "Model and Geometry", ncols = 1, nrows = len(imgs), figsize = (xdim, zdim))

        cmap = mpl.colormaps["jet"]

        if len(imgs) > 1:

            for i in range(len(imgs)):
                
                ax[i].imshow(imgs[i], aspect = "auto", cmap = "jet")

                ax[i].plot(self.x_src_grid, self.z_src_grid*np.ones(self.src_total), "*", color = "white")
                ax[i].plot(self.x_rec_grid, self.z_rec_grid*np.ones(self.rec_total), "v", color = "red")

                ax[i].set_xticks(xloc)
                ax[i].set_xticklabels(xlab)

                ax[i].set_yticks(zloc)
                ax[i].set_yticklabels(zlab)

                ax[i].set_xlabel("X [m]", fontsize = 15)
                ax[i].set_ylabel("Z [m]", fontsize = 15)

                norm = mpl.colors.Normalize(pmin[i], pmax[i])
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes("bottom", size = "10%", pad = 0.6)
                cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), cax = cax, ticks = np.linspace(pmin[i], pmax[i], 5), orientation = "horizontal")
                cbar.ax.set_xticklabels(np.around(np.linspace(pmin[i], pmax[i], 5), decimals = 1))
                cbar.set_label(legends[i], fontsize = 12)

        else:

            ax.imshow(imgs[0], aspect = "auto", cmap = "jet")

            ax.plot(self.x_src_grid, self.z_src_grid*np.ones(self.src_total), "*", color = "black")
            ax.plot(self.x_rec_grid, self.z_rec_grid*np.ones(self.rec_total), "v", color = "red")

            ax.set_xticks(xloc)
            ax.set_xticklabels(xlab)

            ax.set_yticks(zloc)
            ax.set_yticklabels(zlab)

            ax.set_xlabel("X [m]", fontsize = 15)
            ax.set_ylabel("Z [m]", fontsize = 15)

            norm = mpl.colors.Normalize(pmin[0], pmax[0])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size = "5%", pad = 0.6)
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), cax = cax, ticks = np.linspace(pmin[0], pmax[0], 5), orientation = "horizontal")
            cbar.ax.set_xticklabels(np.around(np.linspace(pmin[0], pmax[0], 5), decimals = 1))
            cbar.set_label(legends[0], fontsize = 12)

        fig.tight_layout()
        plt.show()

    def set_modeling_parameters(self):
        
        self.set_parameters()
        self.set_Cerjan_abc()
        self.set_geometry()
        self.set_wavelet()
        self.set_models()
        self.set_wavefield()

    def set_parameters(self):
        
        self.nx = int(self.catch_parameter("x_samples")[0])
        self.nz = int(self.catch_parameter("z_samples")[0])
        self.nt = int(self.catch_parameter("t_samples")[0])

        self.dx = float(self.catch_parameter("x_spacing")[0])
        self.dz = float(self.catch_parameter("z_spacing")[0])
        self.dt = float(self.catch_parameter("t_spacing")[0])

        self.nabc = int(self.catch_parameter("abc_samples")[0])
        self.fabc = float(self.catch_parameter("abc_factor")[0])

        self.fmax = float(self.catch_parameter("max_frequency")[0])

        self.vp_filename = self.catch_parameter("vp_filename")[0]  
        self.vs_filename = self.catch_parameter("vs_filename")[0]  
        self.rho_filename = self.catch_parameter("rho_filename")[0]  

        self.z_src = float(self.catch_parameter("src_elevation")[0])
        self.z_rec = float(self.catch_parameter("rec_elevation")[0])

        self.src_limits = np.array(self.catch_parameter("x_src"), dtype = float)
        self.rec_limits = np.array(self.catch_parameter("x_rec"), dtype = float)

        self.src_total = int(self.src_limits[2])
        self.rec_total = int(self.rec_limits[2])

        self.reciprocity = eval(self.catch_parameter("reciprocity")[0]) 

        self.total_snapshots = int(self.catch_parameter("total_snapshots")[0])

    def set_Cerjan_abc(self):
        ''' Absorbing Boundary Condition: 

            A nonreflecting boundary condition for discrete 
            acoustic and elastic wave equations; Cerjan et al (1985). '''
        
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

        self.x_src = np.linspace(self.src_limits[0], self.src_limits[1], self.src_total)
        self.x_rec = np.linspace(self.rec_limits[0], self.rec_limits[1], self.rec_total)

        if self.reciprocity:

            self.z_src, self.z_rec = self.z_rec, self.z_src
            self.x_src, self.x_rec = self.x_rec, self.x_src

            self.src_total, self.rec_total = self.rec_total, self.src_total

        self.x_src_grid = np.array(self.x_src / self.dx, dtype = int)
        self.x_rec_grid = np.array(self.x_rec / self.dx, dtype = int)

        self.z_src_grid = int(self.z_src / self.dz)
        self.z_rec_grid = int(self.z_rec / self.dz)            
        
    def set_wavelet(self):
        raise NotImplementedError("Please implement this method")

    def set_models(self):
        raise NotImplementedError("Please implement this method")

    def set_wavefield(self):
        raise NotImplementedError("Please implement this method")

    def propagation(self):
        
        ti = time()

        for self.shot_id in range(self.src_total):

            self.initialization()
        
            for self.time_id in range(self.nt):
                
                self.show_information()
                
                self.apply_wavelet()

                self.solve_8E2T_FDM()

                self.apply_boundary_condition()

                self.get_snapshot()

                self.get_seismogram()

        tf = time()

        run_time = tf - ti

        print(f"\nRun time: {run_time:.3f} seconds")

        self.snapshots.flatten("F").astype(np.float32, order = "F").tofile(f"snapshots_Nt{self.total_snapshots}_Nz{self.nz}_Nx{self.nx}_dx{self.dx:.0f}m_dz{self.dz:.0f}m.bin")

    def initialization(self):
        
        self.P = np.zeros((self.nzz, self.nxx))
        
        self.seismogram = np.zeros((self.nt, self.rec_total))
        self.snapshots = np.zeros((self.nz, self.nx, self.total_snapshots))

        self.set_wavefield()

        self.sIdx = self.x_src_grid[self.shot_id] + self.nabc
        self.sIdz = self.z_src_grid + self.nabc

        self.snap_count = 0

    def show_information(self):

        if self.time_id % 100 == 0:    

            system("clear")            
            print(f"Wave propagation in {self.name} isotropic media")
            print(f"\nModel dimensions: (z = {(self.nz-1)*self.dz:.0f}, x = {(self.nx-1)*self.dx:.0f}) meters")
            print(f"\nWavelet max frequency: {self.fmax} Hz")
            print(f"\nRunning shot {self.shot_id+1} of {self.src_total}")        
            print(f"\nPropagation time: {100.0*float(self.time_id+1)/(self.nt):.1f} %")

    def apply_wavelet(self):
        raise NotImplementedError("Please implement this method")    

    def solve_8E2T_FDM(self):
        raise NotImplementedError("Please implement this method")    

    def apply_boundary_condition(self):
        raise NotImplementedError("Please implement this method")    
            
    def get_snapshot(self):

        if (self.time_id) % (self.nt / self.total_snapshots) == 0:
            self.snapshots[:,:,self.snap_count] = self.P[self.nabc:self.nzz-self.nabc,self.nabc:self.nxx-self.nabc] 
            self.snap_count += 1

    def get_seismogram(self):
        pass

    def plot_models(self):
        raise NotImplementedError("Please implement this method")

    def plot_wavelet(self):    

        ft_wavelet = np.fft.fft(self.wavelet)
        freqs = np.fft.fftfreq(self.nt, self.dt)    
        mask = freqs >= 0

        fig, ax = plt.subplots(num = "Wavelet", nrows = 2, ncols = 1, figsize = (10, 4))

        ax[0].plot(np.arange(self.nt)*self.dt, self.wavelet)
        ax[0].set_xlim([0, self.nt*self.dt])
        ax[0].set_xlabel("Time [s]", fontsize = 15)
        ax[0].set_ylabel("Amplitude", fontsize = 15)

        ax[1].stem(freqs[mask], np.abs(ft_wavelet[mask]), markerfmt = "k", linefmt = "k--", basefmt = "k")
        ax[1].set_xlim([0, self.fmax])
        ax[1].set_xlabel("Frequency [Hz]", fontsize = 15)
        ax[1].set_ylabel("Amplitude", fontsize = 15)

        fig.tight_layout()
        plt.show()

    def plot_snapshots(self):
        pass

    def plot_seismogram(self):
        pass

            
