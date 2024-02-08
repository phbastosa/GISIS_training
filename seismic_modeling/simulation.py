from modeling import scalar

from time import time

def main_simulation():

    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

    myWave[id].set_model()
    myWave[id].set_wavelet()    

    beg = time()
    myWave[id].fdm_propagation()
    end = time()

    print(f"\nModeling Run Time = {end - beg:.3f} seconds")

    myWave[id].plot_model()
    myWave[id].plot_wavelet()
    myWave[id].plot_wavefield()
    myWave[id].plot_seismogram()
    myWave[id].plot_wave_propagation()


if __name__ == "__main__":
    main_simulation()


