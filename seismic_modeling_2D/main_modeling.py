from modules.type_scalar import Scalar
from modules.type_elastic import Elastic
from modules.type_acoustic import Acoustic
from modules.base_modeling import BaseModeling

def main():

    myWave2D = [Scalar(), Acoustic(), Elastic()]

    mId = BaseModeling().modeling_type

    myWave2D[mId].set_modeling_parameters()

    myWave2D[mId].plot_models()
    myWave2D[mId].plot_wavelet()

    myWave2D[mId].propagation()

if __name__ == "__main__":
    main()


