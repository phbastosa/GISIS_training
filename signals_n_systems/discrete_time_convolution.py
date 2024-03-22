import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

def discreteTimeConvolution(n1,x1,n2,x2):

    n = np.arange(np.min([n1[0],n2[0]]),len(n1)+len(n2)-4,1)

    matrix = np.zeros((len(n),len(n1)))

    for i in range(len(n)):
        
        if i+1 < len(n1):
            if i+1 < len(n2):
                matrix[i,:i+1] = np.flip(x2[:i+1])
            else:
                matrix[i,i-len(n2)+1:i+1] = np.flip(x2[:])
        else:
            if i+1 < len(n2):
                matrix[i,:] = np.flip(x2[i-len(n1)+1:i+1])
            else:
                matrix[i,i-len(n2)+1:] = np.flip(x2[i-len(n1)+1:])

    y = np.dot(matrix, x1)

    return n, y  

def plot_convolution(n1,x1,n2,x2,n,y):

    k = np.arange(n[0]-len(n2)+2,n[0]+2)

    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (16,8))

    ax[0][0].stem(n1, x1, markerfmt = "g", linefmt = "g--", basefmt = "g")
    ax[0][0].set_xlim([n1[0]-2, n1[-1]+2])
    ax[0][0].set_title("x1[n]")
    ax[0][0].set_xlabel("n")

    ax[1][0].stem(n2, x2, markerfmt = "k", linefmt = "k--", basefmt = "k")
    ax[1][0].set_title("x2[n]")

    ax[0][1].stem(n1, x1, markerfmt = "g", linefmt = "g--", basefmt = "g")
    ax[0][1].set_title("x1[k] * x2[n]")
    ax[0][1].set_xlim(n[0]-len(n2), n[-1])
    ax[0][1].set_xticks(np.arange(n[0]-len(n2), n[-1]+1, dtype = int))
    ax[0][1].set_xticklabels(np.arange(n[0]-len(n2), n[-1]+1, dtype = int))

    ax[1][1].set_ylim(np.min(y), np.max(y))
    ax[1][1].set_title("y[n]", fontsize = 18)
    ax[1][1].set_xlabel("n", fontsize = 15)
    ax[1][1].set_xlim(n[0]-len(n2), n[-1])
    ax[1][1].set_xticks(np.arange(n[0]-len(n2), n[-1]+1, dtype = int))
    ax[1][1].set_xticklabels(np.arange(n[0]-len(n2), n[-1]+1, dtype = int))

    h1 = ax[0][1].stem(k, np.flip(x2), markerfmt = "k", linefmt = "k--", basefmt = "k")
    h2 = ax[1][1].stem(n[0],y[0], markerfmt = "b", linefmt = "b--", basefmt = "b")

    bottom = 0

    def update(i):

        h1[0].set_ydata(np.flip(x2))
        h1[0].set_xdata(k+i)

        h2[0].set_ydata(y[:i+1])
        h2[0].set_xdata(n[:i+1])

        h1[1].set_paths([np.array([[xx, bottom],[xx, yy]]) for (xx, yy) in zip(k+i, np.flip(x2))])
        h2[1].set_paths([np.array([[xx, bottom],[xx, yy]]) for (xx, yy) in zip(n[:i+1], y[:i+1])])

        h1[2].set_xdata([np.min(k+i), np.max(k+i)])
        h2[2].set_xdata([np.min(n[:i+1]), np.max(n[:i+1])])

    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=range(len(n)), interval=500)
    anim.save('discrete_time_convolution.gif', dpi=96, writer='pillow')

    plt.show() 

#----------------------------------------------------------------------

n2 = np.arange(-3,4)
n1 = np.arange(-2,5)

x2 = np.sin(0.25 * np.pi * n2)
x1 = np.exp(-0.9 * n1)

n, y = discreteTimeConvolution(n1,x1,n2,x2)

plot_convolution(n1,x1,n2,x2,n,y)
