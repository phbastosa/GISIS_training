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

def plot_seismic(data : sgy.SegyFile, sort_type : str, index : int) -> None:

    '''
    Plot a common seismic gather.
    
    Parameters
    ----------        
    
    data: segyio object.

    sort_type: ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    Examples
    --------
    >>> plot_seismic(data, sort_type = "src", index = 51)
    >>> plot_seismic(data, sort_type = "rec", index = 203)
    >>> plot_seismic(data, sort_type = "cmp", index = 315)
    >>> plot_seismic(data, sort_type = "off", index = 223750)
    '''    

    match sort_type:

        case "src":
            byte = 9; label = "shot"

        case "rec":
            byte = 13; label = "receiver"

        case "off":
            byte = 37; label = "offset"

        case "cmp":
            byte = 21; label = "mid point"

        case _:
            print("Invalid sort type!")

    print(f"building common {label} gather")        
    if index not in data.attributes(byte)[:]:
        print("Invalid index!")
        exit()

    traces = np.where(data.attributes(byte)[:] == index)[0]

    seismic = data.trace.raw[:].T
    seismic = seismic[:,traces]

    scale = 0.05*np.std(seismic)

    sx = data.attributes(73)[traces]
    sy = data.attributes(77)[traces]

    gx = data.attributes(81)[traces]
    gy = data.attributes(85)[traces]

    cmpx = data.attributes(181)[traces]
    cmpy = data.attributes(185)[traces]


    plt.figure(1)
    plt.imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
    plt.tight_layout()

    plt.figure(2)
    plt.plot(sx/100, sy/100, "o")
    plt.plot(gx/100, gy/100, "o")
    # plt.plot(cmpx/100, cmpy/100, "o")
    plt.show()

# -------------------------------------------------------------------------------

data = sgy.open("2D_Land_vibro_data_2ms/seismic_raw.sgy", ignore_geometry = True)

# show_trace_header(data)

# plot_seismic(data, sort_type = "src", index = 51)
plot_seismic(data, sort_type = "rec", index = 203)
# plot_seismic(data, sort_type = "cmp", index = 315)
# plot_seismic(data, sort_type = "off", index = 223750)

