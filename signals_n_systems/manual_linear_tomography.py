import numpy as np

L = 1000.0       # Model length

# Source and receiver positions
src = np.array([[0.0, 0.5*L], [0.0, 1.5*L], [0.0, 2.5*L]]) 
rec = np.array([[3*L, 0.5*L], [3*L, 1.5*L], [3*L, 2.5*L]])

# Computing angles
angle = np.arctan((rec[:,1] - src[0,1]) / (rec[:,0] - src[0,0]))

# Computing distances
d1 = L

x = L*np.tan(angle[1])

d2 = np.sqrt(x*x + L*L)
d3 = np.sqrt((0.5*L - x)*(0.5*L - x) + 0.25*L*L)

x = 0.5*L*np.tan(0.5*np.pi - angle[2]) 
y = (L - x)*np.tan(angle[2])

d4 = np.sqrt(x*x + 0.25*L*L)
d5 = np.sqrt((L - x)*(L - x) + y*y)
d6 = np.sqrt(L*L + (L - 2*y)*(L - 2*y))

# Distance validation
R2 = 2*d2 + 2*d3
R_2 = np.sqrt(9*L*L + L*L)

R3 = 2*d4 + 2*d5 + d6
R_3 = np.sqrt(9*L*L + 4*L*L)

print(f"Is the distance for the ray 2 ok? {int(R2) == int(R_2)}")
print(f"Is the distance for the ray 3 ok? {int(R3) == int(R_3)}")

# Creating G matrix
M = 9
N = 9

G = np.array([[d1, d1, d1, 0, 0, 0, 0, 0, 0],
              [d2, d3, 0, 0, d3, d2, 0, 0, 0],
              [d4, 0, 0, d5, d6, d5, 0, 0, d4],
              [ 0, d3, d2, d2, d3, 0, 0, 0, 0],
              [ 0, 0, 0, d1, d1, d1, 0, 0, 0],
              [ 0, 0, 0, d2, d3, 0, 0, d3, d2],
              [ 0, 0, d4, d5, d6, d5, d4, 0, 0],
              [ 0, 0, 0, 0, d3, d2, d2, d3, 0],
              [ 0, 0, 0, 0, 0, 0, d1, d1, d1]])

# reading from a text file
m = 1.0 / np.array([1500, 1500, 1500, 
                    1600, 1600, 1600, 
                    1700, 1700, 1700])

# Computing travel times
t = np.dot(G, m)

# write in a text file
np.savetxt("travel_times.txt", t, fmt = "%.6f")