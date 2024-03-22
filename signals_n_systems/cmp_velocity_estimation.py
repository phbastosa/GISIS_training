import numpy as np
import matplotlib.pyplot as plt

nt = 5001
dt = 1e-3

nx = 161
dx = 25.0

seismic = np.fromfile(f"cmp_gather_{nt}x{nx}_{dt*1e6:.0f}us.bin", dtype = np.float32, count = nt*nx)
seismic = np.reshape(seismic, [nt,nx], order = "F")

# for 4 sparse layers

n_layers = 4

G = np.ones((nx, 2))
X = (np.arange(nx)*dx)**2
T = np.zeros((n_layers, nx))

for offset in range(nx):
    max_amplitude = np.max(seismic[:,offset])
    picks = np.where(seismic[:,offset] == max_amplitude)[0]
    dpicks = np.append([0], picks[1:] - picks[:-1]) 
    T[:,offset] = np.delete(picks, np.where(dpicks == 1))*dt

T = T*T
G[:,1] = X
GTG = np.dot(G.T, G)

# equation: t**2 = t0**2 + x**2/vrms**2

t0 = np.zeros(n_layers)
vrms = np.zeros(n_layers)
vint = np.zeros(n_layers)
depth = np.zeros(n_layers)

for i in range(n_layers):
    GTd = np.dot(G.T, T[i,:])        

    m = np.linalg.solve(GTG, GTd)
    
    t0[i] = np.sqrt(m[0])
    vrms[i] = 1.0 / np.sqrt(m[1]) 

vint[0] = vrms[0]
depth[0] = 0.5*t0[0]*vint[0]

for i in range(1, n_layers):
    vint[i] = np.sqrt((vrms[i]**2*t0[i] - vrms[i-1]**2*t0[i-1]) / (t0[i] - t0[i-1]))
    depth[i] = depth[i-1] + 0.5*(t0[i] - t0[i-1])*vint[i]
      
print(t0)
print(vint)
print(depth)

true_vint = np.array([1500, 1650, 2000, 3000, 4500])
true_depth = np.array([500, 1500, 2500, 3500])

z = np.arange(int((np.max(true_depth) + 3000)/dx))*dx

inv_model = vint[0] * np.ones_like(z)
true_model = true_vint[0] * np.ones_like(z) 
for i in range(1, len(true_vint)):        
    true_model[int(true_depth[i-1]/dx):] = true_vint[i]
    inv_model[int(depth[i-1]/dx):] = vint[i] if i < len(vint) else np.nan

 


fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (9, 8))

ax[0].imshow(seismic, aspect = "auto", cmap = "Greys")

for i in range(n_layers):
    ax[0].plot(np.arange(nx), np.sqrt(t0[i]**2 + (np.arange(nx)*dx)**2/vrms[i]**2)/dt)

ax[0].plot(np.zeros(len(t0)), t0/dt, "o")
ax[0].set_xticks(np.linspace(0, nx, 5))
ax[0].set_xticklabels(np.linspace(0, nx-1, 5)*dx)

ax[0].set_yticks(np.linspace(0, nt, 11))
ax[0].set_yticklabels(np.linspace(0, nt-1, 11)*dt)


ax[0].set_title("CMP Gather", fontsize = 18)
ax[0].set_xlabel("x = Offset [m]", fontsize = 15)
ax[0].set_ylabel("t = TWT [s]", fontsize = 15)

ax[1].plot(true_model, z)
ax[1].plot(inv_model, z)

ax[1].set_title("Estimated model", fontsize = 18)
ax[1].set_xlabel("velocities [m/s]", fontsize = 15)
ax[1].set_ylabel("Depth [m]", fontsize = 15)

ax[1].set_ylim([0, z[-1]])
ax[1].invert_yaxis()

fig.tight_layout()
plt.show()

