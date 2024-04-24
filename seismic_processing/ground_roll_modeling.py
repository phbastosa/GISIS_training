import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

def analytical_time_reflections(v, z, x):
    
    Tint = 2.0 * z / v[:-1]
    Vrms = np.zeros(len(z))
    
    reflections = np.zeros((len(z), len(x)))
    for i in range(len(z)):
        Vrms[i] = np.sqrt(np.sum(v[:i+1]**2 * Tint[:i+1]) / np.sum(Tint[:i+1]))
        reflections[i] = np.sqrt(x**2.0 + 4.0*np.sum(z[:i+1])**2) / Vrms[i]
 
    return reflections

def analytical_amps_reflections(v, z):

    reflectivity = np.zeros(len(z))
    reflectivity = (v[1:] - v[:-1]) / (v[1:] + v[:-1]) 
    
    reflections = np.zeros_like(reflectivity)
    transmission = np.zeros_like(reflectivity) 
    
    reflections[0] = reflectivity[0]         
    transmission[0] = 1 - reflectivity[0]     

    for i in range(1, len(reflectivity)):   
        reflections[i] = transmission[i-1] * reflectivity[i]         
        transmission[i] = transmission[i-1] * (1 - reflectivity[i]) 

        for j in range(i, 0, -1):  
            reflections[i] *= (1 - reflectivity[i-j])    

    return reflections

def wavelet_generation(nt, dt, fmax):    
    t0 = 0.5*nt*dt
    fc = fmax / (3.0 * np.sqrt(np.pi)) 
    arg = np.pi*((np.arange(nt)*dt - t0)*fc*np.pi)**2
    return (1.0 - 2.0*arg)*np.exp(-arg)


time_samples = 1001
time_spacing = 1e-3

total_receivers = 81
receiver_spacing = 10.0

source_frequency = 50

frequency_beg = 5.0
frequency_end = 25.0

velocity_end = 100.0
velocity_beg = 250.0

t0 = 0.5

z = np.array([100, 100])
v = np.array([650, 800, 1000])

time = np.arange(time_samples) * time_spacing
offset = np.arange(total_receivers) * receiver_spacing + receiver_spacing

reflections_amps = analytical_amps_reflections(v, z)
reflections_time = analytical_time_reflections(v, z, offset)

wavelet = wavelet_generation(time_samples, time_spacing, source_frequency)

frequencies = frequency_beg + 0.5*(frequency_end - frequency_beg)*time / ((time_samples-1)*time_spacing)
ground_roll_shape = np.sin(2.0*np.pi*frequencies*time)

time_beg = offset/velocity_beg
time_end = offset/velocity_end + t0

cmp_gather = np.zeros((time_samples, total_receivers))

for trace in range(total_receivers):

    direct_wave_time_sample = int(offset[trace]/v[0]/time_spacing)

    if direct_wave_time_sample < time_samples:
        cmp_gather[direct_wave_time_sample,trace] = 1.0    

    for layer in range(len(z)):
        time_index = int(reflections_time[layer,trace] / time_spacing)
        
        if time_index < time_samples:
            cmp_gather[time_index,trace] = reflections_amps[layer]

    cmp_gather[:,trace] = np.convolve(cmp_gather[:,trace], wavelet, "same")

    region = np.logical_and(time_beg[trace] < time, time_end[trace] > time)
    
    tapper_length = len(np.where(region == True)[0]) 

    tapper = signal.windows.tukey(tapper_length)

    tapper[:int(0.5*tapper_length)] = 1.0 

    ground_roll_amplitude = ground_roll_shape[:tapper_length] * tapper

    cmp_gather[region, trace] += 0.8*ground_roll_amplitude 

    cmp_gather[:,trace] *= time[::-1]**1.5
    
    cmp_gather[:,trace] *= offset[trace]**(-0.1) if offset[trace] != 0.0 else 0.9


fk_domain = np.fft.fftn(cmp_gather)

plt.imshow(np.abs(fk_domain), aspect = "auto")
plt.show()


plt.imshow(cmp_gather, aspect = "auto", cmap = "Greys")
plt.show()

cmp_gather.flatten("F").astype(np.float32, order = "F").tofile("cmp_gather_1001x81.bin")