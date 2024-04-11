from sys import argv
from time import time
from modeling import Modeling

scalar = Modeling(argv[1])

scalar.set_parameters()

scalar.import_model()
scalar.set_geometry()

scalar.set_boundary()
scalar.set_wavelet()
scalar.set_damper()

ti = time()
scalar.fdm_propagation()
tf = time()

print(f"\nRuntime = {tf - ti:.3f} s")

scalar.plot_wavelet()
scalar.plot_geometry()
scalar.plot_seismogram()
scalar.plot_snapshots()
