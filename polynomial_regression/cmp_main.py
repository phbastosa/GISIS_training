import functions 

import numpy as np
import matplotlib.pyplot as plt

z_true = 500
v_true = 3000

p_true = [(2*z_true/v_true)**2, (1/v_true)**2]

order = len(p_true)

off_min = 50
off_max = 8050
off_num = 321

offset = np.linspace(off_min, off_max, off_num)

noise_amp = 50e-3
exact_data = np.sqrt(functions.build_polynomial_function(p_true, offset**2))

d_obs = functions.add_noise(exact_data, noise_amp)

p_calc = functions.least_squares_solution(offset**2, order, d_obs**2)

v_calc, z_calc = np.sqrt(1/p_calc[1]), 0.5*np.sqrt(p_calc[0]/p_calc[1])

d_cal = np.sqrt(functions.build_polynomial_function(p_calc, offset**2))


fig, ax = plt.subplots(num = "CMP parameters estimation", figsize = (10,7))

ax.plot(offset, d_obs, label = f"velocity = {v_true:.1f} m/s, depth = {z_true:.1f} m")
ax.plot(offset, d_cal, label = f"velocity = {v_calc:.1f} m/s, depth = {z_calc:.1f} m")

ax.set_xlabel("Offset [m]", fontsize = 15)
ax.set_ylabel("TWT [s]", fontsize = 15)

ax.legend(loc = "upper right", fontsize = 12)

ax.invert_yaxis()

fig.tight_layout()
plt.show()