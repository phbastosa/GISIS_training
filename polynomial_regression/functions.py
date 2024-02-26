import numpy as np

def build_polynomial_function(parameters, x):
	
	polynom = np.zeros_like(x)
	
	for k, p in enumerate(parameters):		
		polynom += p*x**k	
	
	return polynom  

def add_noise(data, noise_amplitude):
	return data + noise_amplitude*(0.5 - np.random.rand(len(data)))

def least_squares_solution(x, order, d):

	G = np.zeros((len(d), order))
	
	for k in range(order): 
		G[:,k] = x**k

	GTG = G.T @ G
	GTd = G.T @ d

	return np.linalg.solve(GTG, GTd) 
