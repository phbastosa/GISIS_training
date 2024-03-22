import numpy as np
import matplotlib.pyplot as plt

receiver_spacing = 25        
receivers_per_shot = 320

total_shots = 161       
near_offset = 25    
shot_spacing = 25      

id = np.zeros(receivers_per_shot*total_shots)
sx = np.zeros(receivers_per_shot*total_shots)
gx = np.zeros(receivers_per_shot*total_shots)

id[:receivers_per_shot] = np.ones(receivers_per_shot)
gx[:receivers_per_shot] = np.arange(receivers_per_shot) * receiver_spacing
sx[:receivers_per_shot] = np.ones(receivers_per_shot) * (receivers_per_shot-1) * receiver_spacing + near_offset  

for i in range(1, total_shots):
    slicer = slice(i*receivers_per_shot, i*receivers_per_shot + receivers_per_shot)

    id[slicer] = i+1
    gx[slicer] = gx[:receivers_per_shot] + i*shot_spacing
    sx[slicer] = sx[:receivers_per_shot] + i*shot_spacing

cmpx = np.array([])
cmpc = np.array([])

cmps = sx - (sx - gx) / 2
for cmp in cmps:
    if cmp not in cmpx:
        cmpx = np.append(cmpx,cmp)
        cmpc = np.append(cmpc,len(np.where(cmp == cmps)[0]))

plt.figure(1,figsize=(10,7))
plt.subplot(211)
plt.plot(gx,id, ".", markersize = 5)
plt.plot(sx,id,".", markersize = 5)
plt.plot(cmpx, np.ones(len(cmpx))*total_shots+10, ".", markersize = 5)
plt.gca().invert_yaxis()
plt.xlim([-5,sx[-1]+5])
plt.gca().set_xticklabels([])
plt.title(f"{receivers_per_shot*total_shots} traces in total", fontsize = 18)
plt.ylabel("Source Id", fontsize = 15)
plt.legend(["Receivers","Sources","Mid points"], loc = "lower left", fontsize = 15)

plt.subplot(212)
plt.stem(cmpx, cmpc)
plt.xlim([-5,sx[-1]+5])
plt.grid(axis="y")
plt.xlabel("Distance [m]", fontsize = 15)
plt.ylabel("CMP traces", fontsize = 15)

plt.tight_layout()
plt.show()
