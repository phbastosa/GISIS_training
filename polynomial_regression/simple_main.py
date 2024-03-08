import numpy as np
import matplotlib.pyplot as plt

from functions import *

# configuration -------------------------------------------

N = 1001

p_true = np.array([0.5, 0.7, -0.2, 0.3])

x = np.linspace(0, 2, N, dtype = float)

# modeling ------------------------------------------------
  
yt = build_polynomial_function(p_true, x)

yn = add_noise(yt, 0.1)

yn[500:700] *= 2

# inversion -----------------------------------------------

p_calc = least_squares_solution(x, len(p_true), yn)

# validation ----------------------------------------------

yf = build_polynomial_function(p_calc, x)

# results -------------------------------------------------

fig, ax = plt.subplots(num = "Parameters estimation", figsize = (10,7))

ax.plot(x, yn, "ok", label = f"Observed data")
ax.plot(x, yt, "-r", label = f"p_true = {np.around(p_true, decimals = 1)}")
ax.plot(x, yf, "-b", label = f"p_calc = {np.around(p_calc, decimals = 1)}")

ax.set_xlabel("X", fontsize = 15)
ax.set_ylabel("Y", fontsize = 15)

ax.legend(loc = "upper right", fontsize = 12)

ax.invert_yaxis()

fig.tight_layout()
plt.show()
