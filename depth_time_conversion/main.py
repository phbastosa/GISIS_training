import numpy as np

from conversions import Conversion

def read_binary_matrix(nz, nx, filename):
    data = np.fromfile(filename, dtype = np.float32, count = nz*nx)
    return np.reshape(data, [nz, nx], order = 'F')

nx = [681, 601, 801]
nz = [141, 161, 181]

total_time = 4.0       # [s]
time_spacing = 1e-3    # [s]
model_spacing = 25.0   # [m]
max_frequency = 30.0   # [Hz] 

marmousi_model = read_binary_matrix(nz[0],nx[0],"models/marmousi_141x681_25m.bin")
salt_dome_model = read_binary_matrix(nz[1],nx[1],"models/salt_dome_161x601_25m.bin")
overthurst_model = read_binary_matrix(nz[2],nx[2],"models/overthrust_181x801_25m.bin")

converter = Conversion()

marmousi_time_section = converter.model_to_time_section(marmousi_model, model_spacing, total_time, time_spacing, max_frequency)
marmousi_depth_section = converter.model_to_depth_section(marmousi_model, model_spacing, total_time, time_spacing, max_frequency)


