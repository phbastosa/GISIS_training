import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.njit(parallel = True)
def parallel_exp(betha, phase, w, sigma, sumQ, sumGamma, wr, t):
 
    for k in nb.prange(len(w)):
        betha[k] = np.exp(-0.5*t*w[k]/sumQ*((w[k] + sigma)/wr)**(-sumGamma))
        phase[k] = np.exp(-complex(0,1)*t*w[k]*(w[k]/wr)**sumGamma)

def wavelet_generator(wavelet_time, time_spacing, max_frequency):

    centroid_frequency = max_frequency / (3.0 * np.sqrt(np.pi))
    
    time = np.linspace(-0.5*wavelet_time, 0.5*wavelet_time, int(wavelet_time/time_spacing) + 1)

    argument = np.pi*(np.pi**2 * centroid_frequency**2 * time**2)
    
    return (1 - 2.0*argument)*np.exp(-argument)    

def forward_constantQ_modeling(trace, time_spacing, Q, sigma = 0.0001):
    
    f = np.fft.fftfreq(len(trace), time_spacing)

    w = 2.0*np.pi*f
    wr = 2.0*np.pi*(1.0/time_spacing)

    fft_trace = np.fft.fft(trace)

    mask = f > 0.0

    nw = len(w[mask])

    matrix = np.ones((len(trace), len(f[mask])), dtype = complex)

    gamma = 1.0 / (np.pi * Q)

    betha = np.zeros(nw, dtype = complex)
    phase = np.zeros(nw, dtype = complex)

    for trace_index in range(len(trace)):

        sumGamma = np.sum(gamma[:trace_index+1]) / (trace_index+1)
        sumQ = np.sum(Q[:trace_index+1]) / (trace_index+1)

        t = trace_index*time_spacing

        parallel_exp(betha, phase, w[mask], sigma, sumQ, sumGamma, wr, t)

        alpha = betha / (betha**2 + sigma**2)
        betha_dagger = alpha / (alpha**2 + sigma)

        matrix[trace_index,:] = 1.0 / nw * betha_dagger * phase 
    
    return np.real(np.dot(matrix, fft_trace[mask]))


if __name__ == "__main__":

    trace_time = 2.0
    wavelet_time = 0.2
    time_spacing = 1e-3
    max_frequency = 50.0

    wavelet = wavelet_generator(wavelet_time, time_spacing, max_frequency)

    trace_samples = int(trace_time / time_spacing)
    
    trace = np.zeros(trace_samples)

    nPicks = int(trace_time/wavelet_time)
    picks = int(wavelet_time/time_spacing)*np.arange(1, nPicks, dtype = int)
    
    trace[picks] = 1.0

    trace = np.convolve(trace, wavelet, mode = "same")

    Q = 100.0 * np.ones(trace_samples)

    synthetic_trace = forward_constantQ_modeling(trace, time_spacing, Q)

    plt.plot(trace)
    plt.plot(synthetic_trace)
    plt.show()