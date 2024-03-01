import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

class Model_to_time_section():

    def __init__(self, velocity_model, sample_spacing, total_time, time_spacing, max_frequency):

        self.dt = time_spacing
        self.dh = sample_spacing
        self.nt = int(total_time / time_spacing)

        self.fmax = max_frequency

        self.model = velocity_model.copy()

        self.nz, self.nx = np.shape(velocity_model)

        self.build_wavelet()
        self.build_reflections()

        self.time_depth_conversion()

        self.build_section()

        self.plot_section()
    
        return self.section

    def build_reflections(self):

        reflectivity = np.zeros_like(self.model)    

        reflectivity[:-1] = (self.model[1:] - self.model[:-1]) / (self.model[1:] + self.model[:-1]) 

        reflections = np.zeros_like(reflectivity)
        transmission = np.zeros_like(reflectivity)  

        reflections[0] = reflectivity[0]         
        transmission[0] = 1 - reflectivity[0]     

        for i in range(1, len(reflectivity)):   
            reflections[i] = transmission[i-1] * reflectivity[i]         
            transmission[i] = transmission[i-1] * (1 - reflectivity[i]) 

            for j in range(i, 0, -1):  
                reflections[i] *= (1 - reflectivity[i-j])    

        self.amp_depth = reflections.copy()

    def build_wavelet(self):
       
        self.wavelet = np.zeros(self.nt)
 
        fc = self.fmax / (3.0 * np.sqrt(np.pi))
        
        s = int(self.nt / 2)
        for i in range(-s,s):
            arg = np.pi*pow(i*self.dt, 2.0)*pow(fc, 2.0)*pow(np.pi, 2.0)
            self.wavelet[i + s] = (1.0 - 2.0*arg)*np.exp(-arg) 

    def time_depth_conversion(self):
  
        t = np.zeros(self.nx, dtype = int)
        self.amp_time = np.zeros((self.nt, self.nx))
    
        for i in range(self.nz):
            t += np.array(2 * (self.dh / self.model[i]) / self.dt, dtype = int)
            for j in range(self.nx):
                if t[j] < self.nt:
                    self.amp_time[t[j],j] = self.amp_depth[i,j]

    def build_section(self): 
        self.section = np.zeros_like(self.amp_time)
        for j in range(self.nx):
            self.section[:,j] = np.convolve(self.wavelet, self.amp_time[:,j], "same")        

    def plot_section(self):
        
        xloc = np.linspace(0,self.nx-1,9, dtype = int)
        xlab = np.array(xloc * self.dh, dtype = int)

        zloc = np.linspace(0,self.nz-1,9, dtype = int)
        zlab = np.array(zloc * self.dh, dtype = int)

        tloc = np.linspace(0,self.nt,9, dtype = int)
        tlab = np.around(tloc * self.dt, decimals = 2)

        fig, ax = plt.subplots(2,1, figsize = (15,8))

        vmin = np.min(self.model)
        vmax = np.max(self.model)

        cmap = mpl.colormaps["jet"]
        norm = mpl.colors.Normalize(vmin*1e-3, vmax*1e-3)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax, ticks = np.linspace(vmin*1e-3, vmax*1e-3, 11), orientation = "horizontal")
        cbar.ax.set_xticklabels(np.around(np.linspace(vmin*1e-3, vmax*1e-3, 11), decimals = 1))
        cbar.set_label("Velocity [km/s]", fontsize = 12)

        ax[0].imshow(self.model, aspect = "auto", cmap = "jet")
        ax[0].set_xticks(xloc)
        ax[0].set_yticks(zloc)
        ax[0].set_xticklabels(xlab)
        ax[0].set_yticklabels(zlab)

        ax[0].set_title("Velocity model", fontsize = 18)
        ax[0].set_xlabel("Distance [m]", fontsize = 15)
        ax[0].set_ylabel("Depth [m]", fontsize = 15)

        vmin = np.min(self.section)
        vmax = np.max(self.section)

        cmap = mpl.colormaps["Greys"]
        norm = mpl.colors.Normalize(vmin, vmax)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax, ticks = np.linspace(vmin, vmax, 11), orientation = "horizontal")
        cbar.ax.set_xticklabels(np.around(np.linspace(vmin, vmax, 11), decimals = 3))
        cbar.set_label("Amplitude", fontsize = 12)

        scale = 3.0 * np.std(self.section)

        ax[1].imshow(self.section, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
        ax[1].set_xticks(xloc)
        ax[1].set_yticks(tloc)
        ax[1].set_xticklabels(xlab)
        ax[1].set_yticklabels(tlab)

        ax[1].set_title("Seismic section", fontsize = 18)
        ax[1].set_xlabel("Distance [m]", fontsize = 15)
        ax[1].set_ylabel("Two way time [s]", fontsize = 15)

        plt.tight_layout()
        plt.show()

class Time_to_depth():
    pass

class Depth_to_time():
    pass