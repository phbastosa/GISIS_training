import numpy as np
import matplotlib.pyplot as plt

# função que descreve a reta
def reta(a0, a1, x):
	y = a0 + a1*x
	return y 

# aplicar o ruido no eixo y
def ruido(y):
	y_n = y + np.random.rand(len(y))
	return y_n

# visualização da reta
def plot_reta(x,y):
	fig, ax = plt.subplots()
	ax.plot(x,y)
	
	fig.tight_layout()	
	plt.show()

# criar espaço solução com varios coeficientes a0 e a1
# correlacionar atraves da norma L2 a diferença
def solution_space(x,y):
	
	n = 1001
	
	a0 = np.linspace(-4,4,n)
	a1 = np.linspace(-5,5,n)
	
	a0, a1 = np.meshgrid(a0,a1)
	
	mat = np.zeros((n,n))
	
	for i in range(n):
		for j in range(n):	
			y_p = a0[i,j] + a1[i,j]*x	
	
			mat[i,j] = np.sqrt(np.sum((y - y_p)**2))
		
	return mat



# plotar o espaço solução
