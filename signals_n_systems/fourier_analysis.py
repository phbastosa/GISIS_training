import numpy as np
import matplotlib.pyplot as plt

nt = 1501
dt = 2e-3

nx = 282
dx = 25.0

seismic = np.fromfile("open_data_seg_poland_vibroseis_2ms_1501x282_shot_1.bin", dtype = np.float32, count = nx*nt)
seismic = np.reshape(seismic, [nt,nx], order = "F")


fft_time = np.fft.fft(seismic, axis = 0)




fk_transform = np.fft.fftshift(np.fft.fft2(seismic))



scale = 0.5*np.std(seismic)

plt.imshow(seismic, aspect = "auto", cmap = "Greys", vmax = scale, vmin = -scale)
plt.show()

scale = 0.5*np.std(np.abs(fft_time))

# abs = sqrt(real**2 + imag**2)
plt.imshow(np.abs(fft_time), aspect = "auto", vmax = scale, vmin = -scale)
plt.show()


f = np.fft.fftfreq(nt, dt)

plt.plot(f, np.abs(fft_time[:,100]))
plt.show()

plt.imshow(np.abs(fk_transform), aspect = "auto")
plt.show()