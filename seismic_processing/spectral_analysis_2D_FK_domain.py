import numpy as np
import segyio as sgy

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

data_input_path = "2D_Land_vibro_data_2ms/seismic_bandpass_20-40Hz.sgy"

shot_gather = 51

data_input = sgy.open(data_input_path, ignore_geometry = True, mode = "r")

traces = np.where(data_input.attributes(9)[:] == shot_gather)[0]

dx = 25.0
nx = len(traces)
nt = data_input.attributes(115)[0][0]                           
dt = data_input.attributes(117)[0][0] / 1e6                     

seismic_input = data_input.trace.raw[:].T

seismic_input = seismic_input[:,traces]

seismic_input_fk = np.fft.fftshift(np.fft.fft2(seismic_input))

frequency = np.fft.fftshift(np.fft.fftfreq(nt, dt))
wavenumber = np.fft.fftshift(np.fft.fftfreq(nx, dx))

df = np.abs(np.abs(frequency[1]) - np.abs(frequency[0]))
dk = np.abs(np.abs(wavenumber[1]) - np.abs(wavenumber[0]))

x = np.arange(nx) 

angle = 60

y1 = np.array(+angle*(np.pi/180)*(x - 0.5*nx) + 0.5*nt, dtype = int) 
y2 = np.array(-angle*(np.pi/180)*(x - 0.5*nx) + 0.5*nt, dtype = int) 

fkfilter = np.ones((nt,nx))

y1 = np.array(y1, dtype = int)
y2 = np.array(y2, dtype = int)

for i in range(nx):
    fill = slice(y2[i],y1[i]) if i > int(0.5*nx) else slice(y1[i],y2[i])    
    fkfilter[fill,i] = 0.0

fkfilter = gaussian_filter(fkfilter, 5)

seismic_output_fk = seismic_input_fk * fkfilter

seismic_output = np.real(np.fft.ifft2(np.fft.ifftshift(seismic_output_fk)))





xloc = np.linspace(0, nx, 5)
xlab = np.around(xloc*dx, decimals = 1) - 0.5*nx*dx

tloc = np.linspace(0, nt, 11, dtype = int)
tlab = np.around(tloc*dt, decimals = 1)

scale_input = 0.1*np.std(seismic_input)

fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (8, 9))

ax[0,0].imshow(seismic_input, aspect = "auto", cmap = "Greys", vmin = -scale_input, vmax = scale_input)
ax[0,0].set_yticks(tloc)
ax[0,0].set_yticklabels(tlab)
ax[0,0].set_xticks(xloc)
ax[0,0].set_xticklabels(xlab)
ax[0,0].set_title(f"Input shot gather number = {shot_gather}")
ax[0,0].set_xlabel("Offset [m]")
ax[0,0].set_ylabel("Two way time [s]")

ax[0,1].imshow(np.abs(seismic_input_fk), aspect = "auto", cmap = "jet", extent = [np.min(wavenumber),(-1)*np.min(wavenumber),np.min(frequency),(-1)*np.min(frequency)])
ax[0,1].plot(x*dk - np.max(wavenumber), y1*df - np.max(frequency),"--k")
ax[0,1].plot(x*dk - np.max(wavenumber), y2*df - np.max(frequency),"--k")
ax[0,1].set_ylim([-60,60])
ax[0,1].set_title(f"Input FK domain")
ax[0,1].set_xlabel(r"Wavenumber [m$^{-1}$]")
ax[0,1].set_ylabel("Frequency [Hz]")


ax[1,0].imshow(seismic_output, aspect = "auto", cmap = "Greys", vmin = -scale_input, vmax = scale_input)
ax[1,0].set_yticks(tloc)
ax[1,0].set_yticklabels(tlab)
ax[1,0].set_xticks(xloc)
ax[1,0].set_xticklabels(xlab)
ax[1,0].set_title(f"Output shot gather number = {shot_gather}")
ax[1,0].set_xlabel("Offset [m]")
ax[1,0].set_ylabel("Two way time [s]")

ax[1,1].imshow(np.abs(seismic_output_fk), aspect = "auto", cmap = "jet", extent = [np.min(wavenumber),(-1)*np.min(wavenumber),np.min(frequency),(-1)*np.min(frequency)])
ax[1,1].plot(x*dk - np.max(wavenumber), y1*df - np.max(frequency),"--k")
ax[1,1].plot(x*dk - np.max(wavenumber), y2*df - np.max(frequency),"--k")
ax[1,1].set_ylim([-60,60])
ax[1,1].set_title(f"Input FK domain")
ax[1,1].set_xlabel(r"Wavenumber [m$^{-1}$]")
ax[1,1].set_ylabel("Frequency [Hz]")

fig.tight_layout()
plt.show()