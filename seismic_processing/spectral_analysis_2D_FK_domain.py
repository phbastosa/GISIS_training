import numpy as np
import segyio as sgy

import matplotlib.pyplot as plt

data_input_path = f"2D_Land_vibro_data_2ms/seismic_bandpass_20-40Hz.sgy"

shot_gather = 51

data_input = sgy.open(data_input_path, ignore_geometry = True, mode = "r")

traces = np.where(data_input.attributes(9)[:] == shot_gather)[0]

dx = 25.0
nx = len(traces)
nt = data_input.attributes(115)[0][0]                           
dt = data_input.attributes(117)[0][0] / 1e6                     

seismic_input = data_input.trace.raw[:].T

seismic_input = seismic_input[:,traces]

seismic_input_fk = np.fft.fftn(seismic_input)
# seismic_input_fk[:,:int(0.5*nx)] = np.flip(seismic_input_fk[:,:int(0.5*nx)], axis = 0)

# frequency = np.fft.fftshift(np.fft.fftfreq(nt, dt))
# wavenumber = np.fft.fftshift(np.fft.fftfreq(nx, dx))

# angle = 0.5*np.pi

# x = np.arange(nx) 
# y1 = angle*(x - 0.5*nx) + 0.5*nt 
# y2 = -angle*(x - 0.5*nx) + 0.5*nt 

# mask = np.logical_and(frequency >= 0, frequency <= 51)

# floc = np.linspace(0, len(frequency[mask])-1, 6, dtype = int)
# flab = np.array(frequency[mask][floc][::-1], dtype = int)

# kloc = np.linspace(0, nx-1, 5, dtype = int)
# klab = np.around(wavenumber[kloc], decimals = 2)[::-1]*(-1)

xloc = np.linspace(0, nx, 5)
xlab = np.around(xloc*dx, decimals = 1) - 0.5*nx*dx

tloc = np.linspace(0, nt, 11, dtype = int)
tlab = np.around(tloc*dt, decimals = 1)

scale_input = 0.5*np.std(seismic_input)

fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (8, 9))

ax[0,0].imshow(seismic_input, aspect = "auto", cmap = "Greys", vmin = -scale_input, vmax = scale_input)
ax[0,0].set_yticks(tloc)
ax[0,0].set_yticklabels(tlab)
ax[0,0].set_xticks(xloc)
ax[0,0].set_xticklabels(xlab)
ax[0,0].set_title(f"Input shot gather number = {shot_gather}")
ax[0,0].set_xlabel("Offset [m]")
ax[0,0].set_ylabel("Two way time [s]")



ax[0,1].imshow(np.abs(seismic_input_fk), aspect = "auto", cmap = "jet")
# ax[0,1].set_yticks(floc)
# ax[0,1].set_yticklabels(flab)
# ax[0,1].set_xticks(kloc)
# ax[0,1].set_xticklabels(klab)
# ax[0,1].set_title(f"Input FK domain")
# ax[0,1].set_xlabel(r"Wavenumber [m$^{-1}$]")
# ax[0,1].set_ylabel("Frequency [Hz]")


fig.tight_layout()
plt.show()