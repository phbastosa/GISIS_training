from modeling import scalar

from time import time

def main_simulation():

    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

    myWave[id].set_model()
    myWave[id].set_wavelet()    

    start = time()
    myWave[id].wave_propagation()
    end = time()

    print(end - start)

    myWave[id].plot_model()
    myWave[id].plot_wavelet()
    myWave[id].plot_wavefield()


if __name__ == "__main__":
    main_simulation()


