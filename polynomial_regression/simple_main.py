import numpy as np
import matplotlib.pyplot as plt

from functions import *

# configuration 
N = 1001

p_true = np.array([0.5, 0.7])

x = np.linspace(0, 2, N, dtype = float)

# modeling 
y = build_polynomial_function(p_true, x)

yn = add_noise(y, 0.5)

yn[500:700] += 0.5

# inversion
p_calc = least_squares_solution(x, len(p_true), yn)

# validation
yf = build_polynomial_function(p_calc, x)

# results
fig, ax = plt.subplots(num = "Parameters estimation", figsize = (10,7))

ax.plot(x, yn, "ok", label = f"True: p0 = {p_true[0]:.1f}, p1 = {p_true[1]:.1f}")
ax.plot(x, yf, "-b", label = f"Calc: p0 = {p_calc[0]:.1f}, p1 = {p_calc[1]:.1f}")

ax.set_xlabel("X", fontsize = 15)
ax.set_ylabel("Y", fontsize = 15)

ax.legend(loc = "upper right", fontsize = 12)

ax.invert_yaxis()

fig.tight_layout()
plt.show()
