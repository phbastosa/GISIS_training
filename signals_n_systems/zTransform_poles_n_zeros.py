import numpy as np
import matplotlib.pyplot as plt

from matplotlib import patches
from scipy import signal

# y[n] - 0.5*y[n-2] = x[n] + 3*x[n-1] - x[n-2] 
 
a = np.array([1.0, 0.0, -0.5])
b = np.array([1.0, 3.0, -1.0])

z, p, k = signal.tf2zpk(b,a)

fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (5,5))

uc = patches.Circle((0,0), radius = 1, fill = False, color = 'black', ls = 'dashed', alpha = 0.2)

ax.add_patch(uc)

t1 = plt.plot(z.real, z.imag, 'go', ms=10)
plt.setp(t1, markersize = 10.0, markeredgewidth = 1.0, markeredgecolor = 'k', markerfacecolor = 'g')

t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
plt.setp(t2, markersize = 12.0, markeredgewidth = 3.0, markeredgecolor = 'r', markerfacecolor = 'r')

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.text(1.2, -0.15, "Re", weight = 'bold')
ax.text(-0.15, 1.2, "Im", weight = 'bold')

r = 1.5; plt.axis('scaled'); 
ticks = [-1, -.5, .5, 1]; 

plt.axis([-r, r, -r, r])
plt.xticks(ticks); 
plt.yticks(ticks)

plt.legend(["unit circle","zeros", "poles"])
plt.tight_layout()
plt.show()
