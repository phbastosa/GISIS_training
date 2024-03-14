from modules.wave_solvers_2D import BaseModeling, Scalar, Acoustic, Elastic

def main():

    myWave2D = [Scalar(), Acoustic(), Elastic()]

    mId = BaseModeling().modeling_type

    myWave2D[mId].set_modeling_parameters()

    myWave2D[mId].plot_models()
    myWave2D[mId].plot_wavelet()

if __name__ == "__main__":
    main()


