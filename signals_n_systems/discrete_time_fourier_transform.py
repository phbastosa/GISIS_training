import numpy as np
import matplotlib.pyplot as plt

def delta(n, delay):
	x = np.zeros_like(n)
	for i,k in enumerate(n):
		x[i] = 1.0 if k == -delay else 0.0
	return x
  
def heaviside(n, delay):
    x = np.zeros_like(n)
    for i,k in enumerate(n):
        x[i] = 1.0 if k >= -delay else 0.0
    return x  
  
def discreteTimeFourierTransform(n,x):
    j = complex(0,1)
    w = np.linspace(-np.pi, np.pi, 1001)
    X = np.zeros_like(w, dtype = complex)
    for i,k in enumerate(n):
        X += x[i] * np.exp(-j*w*k)
    return w,X  

#-----------------------------------------------------------------------

n = np.arange(-5,6)

x = heaviside(n,3) - heaviside(n,-1) - 1.0*delta(n,-2) - 0.5*delta(n,-4)

w, X = discreteTimeFourierTransform(n, x)

angles = np.linspace(-180, 180, 5, dtype = int)
angle_locations = np.linspace(-np.pi, np.pi, 9) 
angle_labels = [r'$-\pi$', r'$-\dfrac{2\pi}{3}$', r'$-\dfrac{\pi}{2}$', r'$-\dfrac{\pi}{4}$', r'$0$',r'$\dfrac{\pi}{4}$', r'$\dfrac{\pi}{2}$', r'$\dfrac{2\pi}{3}$', r'$\pi$']

fig, ax = plt.subplots(ncols = 1, nrows = 3, figsize = (10,8))

ax[0].stem(n, x)
ax[0].set_xlim([n[0], n[-1]])
ax[0].set_xticks(n)
ax[0].set_xticklabels(n)

ax[0].set_title("x[n]", fontsize = 18)
ax[0].set_xlabel("n", fontsize = 15)
ax[0].set_ylabel('Amplitude', fontsize = 15)

ax[1].plot(w, np.abs(X))
ax[1].set_xticks(angle_locations)
ax[1].set_xticklabels(angle_labels)
ax[1].set_title(r"$|X(jw)|$", fontsize = 18)
ax[1].set_xlabel('Angular frequency [rad/s]', fontsize = 15)
ax[1].set_ylabel('Amplitude', fontsize = 15)

ax[2].plot(w, 180.0*np.angle(X)/np.pi, ".")
ax[2].set_xticks(angle_locations)
ax[2].set_xticklabels(angle_labels)

ax[2].set_yticks(angles)
ax[2].set_yticklabels(angles)

ax[2].set_title(r"$\angle{X(jw)}$", fontsize = 18)
ax[2].set_xlabel('Angular frequency [rad/s]', fontsize = 15)
ax[2].set_ylabel('Angle [°]', fontsize = 15)

plt.tight_layout()
plt.savefig("discrete_time_fourier_transform.png", dpi = 200)
plt.show()