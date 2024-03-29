import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

cf1 = 10     # continum frequency of signal 1
cf2 = 40     # continum frequency of signal 2

sf = 200     # sampling frequency

w1 = 2.0*np.pi*cf1/sf  # discrete frequency of signal 1 
w2 = 2.0*np.pi*cf2/sf  # discrete frequency of signal 2

ns = int(2.0*np.pi/w1 * 2.0*np.pi/w2)  # computing total samples in fundamental period

n = np.arange(ns)

x1 = np.sin(w1*n)
x2 = np.cos(w2*n)

plt.plot(n, x1)
plt.plot(n, x2)
plt.show()

# Convolution in time domain

yt = signal.convolve(x1, x2, "same")

# Convolution in frequency domain

yf = signal.fftconvolve(x1, x2, "same")

plt.plot(yt)
plt.plot(yf)

plt.show()


plt.stem(np.abs(np.fft.fft(x1)) / ns)
plt.stem(np.abs(np.fft.fft(x2)) / ns)
plt.show()

