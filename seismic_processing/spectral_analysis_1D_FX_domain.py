import numpy as np
import segyio as sgy

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

def butter_bandpass_filter(input_trace, lowcut, highcut, fs, order = 5):
    b, a = butter(order, [lowcut, highcut], fs = fs, btype = 'band')
    return lfilter(b, a, input_trace)

data_path = "2D_Land_vibro_data_2ms/seismic_raw.sgy"

shot_gather = 1

lowcut = 20
highcut = 40

data_output_path = f"2D_Land_vibro_data_2ms/seismic_bandpass_{lowcut:.0f}-{highcut:.0f}Hz.sgy"

data_input = sgy.open(data_path, ignore_geometry = True, mode = "r")

traces = np.where(data_input.attributes(9)[:] == shot_gather)[0]

dx = 25.0 
nx = len(traces)
nt = data_input.attributes(115)[0][0]                           
dt = data_input.attributes(117)[0][0] / 1e6                     

seismic_input = data_input.trace.raw[:].T

seismic_output = np.zeros_like(seismic_input)

for i in range(len(seismic_input[0])):
    seismic_output[:,i] = butter_bandpass_filter(seismic_input[:,i], lowcut, highcut, 1.0/dt)

seismic_input_fft = np.fft.fft(seismic_input, axis = 0)
seismic_output_fft = np.fft.fft(seismic_output, axis = 0)

for i in range(len(seismic_input[0])):

    seismic_input_fft[:,i] *= 1.0 / np.max(np.abs(seismic_input_fft[:,i]))
    seismic_output_fft[:,i] *= 1.0 / np.max(np.abs(seismic_output_fft[:,i]))

sgy.tools.from_array2D(data_output_path, seismic_output.T)
data_output = sgy.open(data_output_path, "r+", ignore_geometry = True)
data_output.header = data_input.header
data_output.close()

seismic_input = seismic_input[:,traces]
seismic_output = seismic_output[:,traces]

seismic_input_fft = seismic_input_fft[:,traces]
seismic_output_fft = seismic_output_fft[:,traces]

frequency = np.fft.fftfreq(nt, dt)

mask = np.logical_and(frequency >= 0, frequency <= 101)

floc = np.linspace(0, len(frequency[mask])-1, 11, dtype = int)
flab = np.array(frequency[floc], dtype = int)

tloc = np.linspace(0, nt-1, 11, dtype = int)
tlab = np.around(tloc*dt, decimals = 1)

xloc = np.linspace(0, nx, 5)
xlab = np.around(xloc*dx, decimals = 1) - 0.5*nx*dx

scale_input = 0.5*np.std(seismic_input)
scale_output = 0.5*np.std(seismic_output)

fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (8, 9))

ax[0,0].imshow(seismic_input, aspect = "auto", cmap = "Greys", vmin = -scale_input, vmax = scale_input)
ax[0,0].set_yticks(tloc)
ax[0,0].set_yticklabels(tlab)
ax[0,0].set_xticks(xloc)
ax[0,0].set_xticklabels(xlab)
ax[0,0].set_title(f"Input shot gather number = {shot_gather}")
ax[0,0].set_xlabel("Offset [m]")
ax[0,0].set_ylabel("Two way time [s]")

ax[0,1].imshow(np.abs(seismic_input_fft[mask,:]), aspect = "auto", cmap = "jet")
ax[0,1].set_yticks(floc)
ax[0,1].set_yticklabels(flab)
ax[0,1].set_xticks(xloc)
ax[0,1].set_xticklabels(xlab)
ax[0,1].set_title(f"Input normalized spectra per trace")
ax[0,1].set_xlabel("Offset [m]")
ax[0,1].set_ylabel("Frequency [Hz]")

ax[1,0].imshow(seismic_output, aspect = "auto", cmap = "Greys", vmin = -scale_output, vmax = scale_output)
ax[1,0].set_yticks(tloc)
ax[1,0].set_yticklabels(tlab)
ax[1,0].set_xticks(xloc)
ax[1,0].set_xticklabels(xlab)
ax[1,0].set_title(f"Output shot gather number = {shot_gather}")
ax[1,0].set_xlabel("Offset [m]")
ax[1,0].set_ylabel("Two way time [s]")

ax[1,1].imshow(np.abs(seismic_output_fft[mask,:]), aspect = "auto", cmap = "jet")
ax[1,1].set_yticks(floc)
ax[1,1].set_yticklabels(flab)
ax[1,1].set_xticks(xloc)
ax[1,1].set_xticklabels(xlab)
ax[1,1].set_title(f"Output normalized spectra per trace")
ax[1,1].set_xlabel("Offset [m]")
ax[1,1].set_ylabel("Frequency [Hz]")

fig.tight_layout()
plt.show()

