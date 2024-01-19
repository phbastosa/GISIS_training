from modeling import scalar

def main_simulation():

    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

    myWave[id].get_type()

    myWave[id].set_wavelet()
    myWave[id].plot_wavelet()

    # myWave[id].set_layer_cake_model()
    # myWave[id].plot_model()

if __name__ == "__main__":
    main_simulation()


