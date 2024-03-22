import numpy as np
import matplotlib.pyplot as plt

def objective_function(x, parameters):
    
    function = 0.0
    for n, p in enumerate(parameters):
        function += p*x**n              
    
    return function                     # a0 + a1.x + a2.x² + a3.x³ + ...

def least_squares_solver(x, d, M):
    
    G = np.zeros((len(d), M))

    for n in range(M): 
        G[:,n] = x**n

    GTG = np.dot(G.T, G)
    GTd = np.dot(G.T, d)

    return np.linalg.solve(GTG, GTd)    # solution of a linear system

#---------------------------------------------------------

m_true = np.array([0.5, 0.2, 2.3])

noise_amp = 0.05

N = 1001

xi = 1.0
xf = 3.0

x = np.linspace(xi, xf, N)

data_true = objective_function(x, m_true)

data_noise = data_true + noise_amp*np.random.randn(N)

# Inversion crime
# m_calc = least_squares_solver(x, data_true, len(true))  

# Outliers
# data_noise[500:600:25] += 2    

m_calc = least_squares_solver(x, data_noise, len(m_true))

data_calc = objective_function(x, m_calc)

print(f"     True parameters: {m_true}")
print(f"Estimated parameters: {m_calc}")

fi, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (10,5))

ax.plot(x, data_true)
ax.plot(x, data_noise, "o", markersize = 3)
ax.plot(x, data_calc)

ax.grid(True)

plt.tight_layout()
plt.show()
