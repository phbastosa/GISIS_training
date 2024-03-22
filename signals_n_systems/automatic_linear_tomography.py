import numpy as np 

def linear_ray_tracing(sId, model, sources, receivers, data, G):

    max_ray_length = 501

    ray_file = open(f"/content/drive/MyDrive/inversao_geofisica_2022/trabalho_final/rays/shot_{sId+1}.txt", "w")

    x = np.linspace(sources.x[0], receivers.x[0], max_ray_length)

    for rayId in range(receivers.n):
        
        m = (sources.z[sId] - receivers.z[rayId]) / (sources.x[sId] - receivers.x[rayId])
        
        z = m * (x - sources.x[sId]) + sources.z[sId]

        ray_step = np.sqrt((x[1] - x[0])**2 + (z[1] - z[0])**2)

        for p in range(max_ray_length):
            
            idx = int(np.floor(x[p] / model.dx))
            idz = int(np.floor(z[p] / model.dz))
            
            G[int(sId * receivers.n + rayId), int(idz + idx * model.nz)] += ray_step

            # data[sId * receivers.n + rayId] += ray_step * bilinear_interpolation(x[p], z[p], model, model.S)
            data[sId * receivers.n + rayId] += ray_step * model.S[idz + model.nb, idx + model.nb]

            ray_file.write(f"{x[p]:.2f}, {z[p]:.2f}\n")


    ray_file.close()

    return None        

