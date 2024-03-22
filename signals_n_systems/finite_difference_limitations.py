import numpy as np
import matplotlib.pyplot as plt

n = 1001
i = complex(0,1)

w = np.linspace(-np.pi, np.pi, n)

d2f = -w**2

fdm_2order = np.exp(i*w) + np.exp(-i*w) - 2.0
fdm_4order =  (16.0*(np.exp(-i*w) + np.exp(i*w)) - (np.exp(-2.0*i*w) + np.exp(2.0*i*w)) - 30.0) / 12.0
fdm_6order =  (270.0*(np.exp(-i*w) + np.exp(i*w)) - 27.0*(np.exp(-2.0*i*w) + np.exp(2.0*i*w)) + 2.0*(np.exp(-3.0*i*w) + np.exp(3.0*i*w)) - 490.0) / 180.0
fdm_8order =  (8064.0*(np.exp(-i*w) + np.exp(i*w)) - 1008.0*(np.exp(-2.0*i*w) + np.exp(2.0*i*w)) + 128.0*(np.exp(-3.0*i*w) + np.exp(3.0*i*w)) - 9.0*(np.exp(-4.0*i*w) + np.exp(4.0*i*w)) - 14350) / 5040.0

angle_locations = np.linspace(-np.pi, np.pi, 9) 
angle_labels = [r'$-\pi$', r'$-\dfrac{2\pi}{3}$', r'$-\dfrac{\pi}{2}$', r'$-\dfrac{\pi}{4}$', r'$0$',r'$\dfrac{\pi}{4}$', r'$\dfrac{\pi}{2}$', r'$\dfrac{2\pi}{3}$', r'$\pi$']

fig, ax = plt.subplots(figsize = (12,5))

ax.plot(w, d2f, label = r'Analytical $\dfrac{d^2f}{dx^2}$' )
ax.plot(w, np.real(fdm_2order), label = 'fdm 2º order')
ax.plot(w, np.real(fdm_4order), label = 'fdm 4º order')
ax.plot(w, np.real(fdm_6order), label = 'fdm 6º order')
ax.plot(w, np.real(fdm_8order), label = 'fdm 8º order')

ax.set_xlim([-np.pi, np.pi])

ax.set_xlabel('Angular frequency [rad/s]', fontsize = 15)
ax.set_ylabel('Amplitude', fontsize =15)

ax.set_xticks(angle_locations)
ax.set_xticklabels(angle_labels)

ax.legend(loc = "upper left")
plt.tight_layout()
plt.savefig("finite_difference_limitations.png", dpi = 200)
plt.show()