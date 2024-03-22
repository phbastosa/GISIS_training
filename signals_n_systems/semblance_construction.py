import numpy as np
import matplotlib.pyplot as plt

nt = 5001
dt = 1e-3

nx = 161
dx = 25.0

vi = 1000
vf = 3000
dv = 50

filename = f"cmp_gather_{nt}x{nx}_{dt*1e6:.0f}us.bin"

seismic = np.fromfile(filename, dtype = np.float32, count = nt*nx)
seismic = np.reshape(seismic, [nt,nx], order = "F")

vrms = np.arange(vi, vf + dv, dv)
offset = np.arange(nx, dtype = int)

time = np.arange(nt) * dt

semblance = np.zeros((nt, len(vrms)))

for indt, t0 in enumerate(time):
    for indv, v in enumerate(vrms):
    
        target = np.array(np.sqrt(t0**2 + (offset*dx/v)**2) / dt, dtype = int) 

        mask = target < nt
    
        t = target[mask]
        x = offset[mask]
    
        semblance[indt, indv] = np.sum(np.abs(seismic[t,x]))**2
    

xloc = np.linspace(0, len(vrms)-1, 9)
xlab = np.linspace(vi, vf, 9)

tloc = np.linspace(0, nt, 11)
tlab = np.around(np.linspace(0, nt-1, 11)*dt, decimals = 3)

scale = 15.0*np.std(semblance)

fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10,8))

ax[0].imshow(seismic, aspect = "auto", cmap = "Greys")
ax[0].set_yticks(tloc)
ax[0].set_yticklabels(tlab)

ax[0].set_xticks(np.linspace(0,nx,5))
ax[0].set_xticklabels(np.linspace(0,nx-1,5, dtype = int)*dx)

ax[0].set_title("CMP Gather", fontsize = 18)
ax[0].set_xlabel("Offset [m]", fontsize = 15)
ax[0].set_ylabel("Two Way Time [s]", fontsize = 15)

ax[1].imshow(semblance, aspect = "auto", cmap = "jet", vmin = -scale, vmax = scale)

ax[1].set_xticks(xloc)
ax[1].set_xticklabels(xlab*1e-3)

ax[1].set_yticks(tloc)
ax[1].set_yticklabels(tlab)

ax[1].set_title("Semblance", fontsize = 18)
ax[1].set_xlabel("RMS Velocity [km/s]", fontsize = 15)
ax[1].set_ylabel("Two Way Time [s]", fontsize = 15)

fig.tight_layout()
plt.grid()
plt.show()
