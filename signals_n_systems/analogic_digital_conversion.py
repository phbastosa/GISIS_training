import numpy as np
import matplotlib.pyplot as plt

sine_frequencies = [10]

total_time = 0.5          # [s]
sampling_frequency = 100  # [Hz]

analogic_time = np.linspace(0, total_time, int(1e6))
analogic_sine = np.zeros_like(analogic_time)

for frequency in sine_frequencies:
    analogic_sine += np.sin(2.0 * np.pi * frequency * analogic_time)  

''' 
    Analytical representation of a digital signal

    w_s = 2 pi f_0 / f_s

    f_0 -> analogic signal frequency
    f_s -> sampling frequency
    w_s -> discrete signal angular frequency
'''

f_0 = np.lcm.reduce(sine_frequencies)  # frequencies mmc
f_s = sampling_frequency

w_s = 2.0*np.pi*f_0/f_s                # discrete angular frequency

dt = 1.0 / f_s                         # time spacing
nt = int(total_time / dt)              # total samples 

n = np.arange(nt)                      # samples array
t = n*dt                               # discrete time array

digital_sine = np.sin(w_s * n)

digital_sine_fft = np.fft.fft(digital_sine)

f = np.fft.fftfreq(nt, dt)             # frequencies array

Nyquist = 0.5*f_s                      # Max frequnecy of a spectrum

fig, ax = plt.subplots(ncols = 1, nrows = 3, figsize = (15, 9))

ax[0].plot(analogic_time, analogic_sine)
ax[0].stem(t, digital_sine, markerfmt = "k", linefmt = "k--", basefmt = "k")
ax[0].set_xlim([0, total_time])
ax[0].set_title("Analogic - Digital Conversion", fontsize = 18)
ax[0].set_xlabel("Time [s]", fontsize = 15)
ax[0].set_ylabel(r"$x(t)$", fontsize = 15)

ax[1].plot(n, digital_sine, "--o")
ax[1].set_xlim([0, nt-1])
ax[1].set_title("Discrete Signal", fontsize = 18)
ax[1].set_xlabel(r"$n$", fontsize = 15)
ax[1].set_ylabel(r"$x[n]$", fontsize = 15)

ax[2].stem(f, np.abs(digital_sine_fft))
ax[2].set_xlim([-Nyquist+1,Nyquist-1])
ax[2].set_title("Discrete Fourier Transform - FFT", fontsize = 18)
ax[2].set_xlabel("Frequency [Hz]", fontsize = 15)
ax[2].set_ylabel(r"$X(f)$", fontsize = 15)

fig.tight_layout()
plt.show()
