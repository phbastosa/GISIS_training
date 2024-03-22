import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors

def objective_function(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2 

def analytical_gradient(x, y):

    dx = 2*(1.5 - x + x*y)*(y - 1) + 2*(2.25 - x + x*y**2)*(y*y - 1) + 2*(2.625 - x + x*y**3)*(y**3 - 1) 
    dy = 2*(1.5 - x + x*y)*x       + 4*(2.25 - x + x*y**2)*x*y       + 6*(2.625 - x + x*y**3)*x*y**2

    return dx, dy

def normalized_gradient(dx, dy):
    
    norm = np.sqrt(dx**2 + dy**2)
    
    dxn = dx / norm
    dyn = dy / norm

    return dxn, dyn

def grid_points(x, y, spacing, total_points):

    xg = x / spacing + 0.5*total_points
    yg = y / spacing + 0.5*total_points

    return xg, yg

#------------------------------------------------------------------------------------------------

limits = [-5, 5]
total_points = 1001

func_domain = np.linspace(limits[0], limits[1], total_points)

spacing = func_domain[1] - func_domain[0]

#------------------------------------------------------------------------------------------------

x, y = np.meshgrid(func_domain, func_domain)

Beale_function = objective_function(x, y)

xm = 3.0   # global minimum
ym = 0.5   # global minimum

#------------------------------------------------------------------------------------------------

xi = 4.0   # initial position  
yi = 2.0   # initial position 

dx, dy = analytical_gradient(xi, yi)
dx, dy = normalized_gradient(-dx,-dy)

y_line_search = dy / dx * (func_domain - xi) + yi

mask = np.logical_and(y_line_search >= limits[0], y_line_search <= limits[1]) 

step_domain = np.arange(len(y_line_search[mask]))
f_line_search = objective_function(func_domain[mask], y_line_search[mask])

best_index = np.where(f_line_search == np.min(f_line_search))[0][0]
init_index = np.where(f_line_search == objective_function(xi, yi))[0][0]

xmg, ymg = grid_points(xm, ym, spacing, total_points)
xig, yig = grid_points(xi, yi, spacing, total_points)
xls, yls = grid_points(func_domain, y_line_search, spacing, total_points)

#------------------------------------------------------------------------------------------------

loc = np.linspace(0, len(func_domain) - 1, 11)
lab = np.linspace(limits[0], limits[1], 11)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,6))

img = ax[0].imshow(Beale_function, aspect = "auto", cmap = "seismic", norm = colors.LogNorm())

cbar = plt.colorbar(img)
cbar.set_label("Beale function", fontsize = 15)

ax[0].set_xlabel("X", fontsize = 15)
ax[0].set_ylabel("Y", fontsize = 15)

ax[0].plot(xig, yig, "ob", label = "Initial position")
ax[0].plot(xmg, ymg, "ok", label = "Global minimum")

ax[0].quiver(xig, yig, dx, dy, color = "green", width = 0.005, scale = 15, label = r"Search direction $-\nabla f(x,y)$")

ax[0].plot(xls[mask], yls[mask], "--b", alpha = 0.4, label = "Step length domain")

ax[0].set_xticks(loc)
ax[0].set_xticklabels(lab)

ax[0].set_yticks(loc)
ax[0].set_yticklabels(lab)

ax[0].invert_yaxis()
ax[0].grid(True)
ax[0].legend(loc = "lower left", fontsize = 10)

ax[1].semilogy(step_domain, f_line_search)
ax[1].plot(init_index, f_line_search[init_index], "ob", label = "Initial position")
ax[1].plot(best_index, f_line_search[best_index], "og", label = "Best update position")

ax[1].set_xticks(step_domain[::10])
ax[1].set_xticklabels(init_index - step_domain[::10])

ax[1].set_xlabel("Step length domain", fontsize = 15)
ax[1].set_ylabel("Beale function", fontsize = 15)
ax[1].legend(loc = "lower left", fontsize = 10)

fig.tight_layout()

plt.show()