import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

def show_binary_header(data):
    binHeader = sgy.binfield.keys
    print("\n Checking binary header \n")
    print(f"{'key': >25s} {'byte': ^6s} {'value': ^7s} \n")
    for k, v in binHeader.items():
        if v in data.bin:
            print(f"{k: >25s} {str(v): ^6s} {str(data.bin[v]): ^7s}")

def show_trace_header(data):
    traceHeader = sgy.tracefield.keys
    print("\n Checking trace header \n")
    print(f"{'Trace header': >40s} {'byte': ^6s} {'first': ^11s} {'last': ^11s} \n")
    for k, v in traceHeader.items():
        first = data.attributes(v)[0][0]
        last = data.attributes(v)[data.tracecount-1][0]
        print(f"{k: >40s} {str(v): ^6s} {str(first): ^11s} {str(last): ^11s}")

def get_traces(attn, data, index, type):
    
    print(f"building common {type} gather")        

    if index not in data.attributes(attn)[:]:
        print("Invalid index!")
        exit()

    return np.where(data.attributes(attn)[:] == index)[0]

def plot_seismic(data, key, index):
    
    seismic = data.trace.raw[:].T

    match key:
        
        case "src":

            traces = get_traces(9, data, index, "shot")

        case "rec":

            traces = get_traces(13, data, index, "receiver")

        case "offset":

            traces = get_traces(37, data, index, "offset")

        case "cmp":

            traces = get_traces(21, data, index, "mid point")

        case _:
            print("Invalid keyword!")

     
    scale = 0.05*np.std(seismic[:,traces])

    plt.imshow(seismic[:,traces], aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------------

data = sgy.open("2D_Land_vibro_data_2ms/seismic_raw.sgy", ignore_geometry = True)

# offset = data.attributes(37)[:282]

# print(offset)

plot_seismic(data, key = "offset", index = 203750)

