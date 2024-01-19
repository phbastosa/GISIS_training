from modeling import scalar

def main_simulation():

    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

    myWave[id].get_type()
    print(myWave[id].type)
    print(scalar.Wavefield_1D.type)

    myWave[id].set_wavelet()
    myWave[id].plot_wavelet()



if __name__ == "__main__":
    main_simulation()


