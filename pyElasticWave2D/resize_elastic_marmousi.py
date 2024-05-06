import numpy as np
import segyio as sgy

data = sgy.open("models/MODEL_P-WAVE_VELOCITY_1.25m.segy", ignore_geometry = True)
vp_raw = data.trace.raw[:].T
vp_new = vp_raw[::4,::4]
vp_new.flatten("F").astype(np.float32, order = "F").tofile(f"models/marmousi_vp_{len(vp_new)}x{len(vp_new[0])}_5m.bin")

data = sgy.open("models/MODEL_S-WAVE_VELOCITY_1.25m.segy", ignore_geometry = True)
vs_raw = data.trace.raw[:].T
vs_new = vs_raw[::4,::4]
vs_new.flatten("F").astype(np.float32, order = "F").tofile(f"models/marmousi_vs_{len(vp_new)}x{len(vp_new[0])}_5m.bin")

data = sgy.open("models/MODEL_S-WAVE_VELOCITY_1.25m.segy", ignore_geometry = True)
rho_raw = data.trace.raw[:].T
rho_new = rho_raw[::4,::4]
rho_new.flatten("F").astype(np.float32, order = "F").tofile(f"models/marmousi_rho_{len(vp_new)}x{len(vp_new[0])}_5m.bin")

