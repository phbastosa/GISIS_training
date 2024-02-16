import functions

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2,2,101)

a0 = -2
a1 = -1

y = functions.reta(a0, a1, x)
yn = functions.ruido(y)

functions.plot_reta(x,yn)

mat = functions.solution_space(x,y)

a0_ind, a1_ind = np.where(mat == np.min(mat))

print(a0_ind, a1_ind)

plt.imshow(mat, extent = [-5,5,-5,5])
plt.show()
