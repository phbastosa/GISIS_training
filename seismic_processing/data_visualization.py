import numpy as np
import segyio as sgy

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


def plot_seismic(data, key, index):
    
    match key:
        
        case "src":
            print("build common shot gather")        

        case "rec":
            print("build common receiver gather")

        case "offset":
            print("build common offset gather")

        case "cmp":
            print("build common mid point gather")

        case _:
            print("Invalid keyword!")


# -------------------------------------------------------------------------------

data = sgy.open("2D_Land_vibro_data_2ms/seismic_raw.sgy", ignore_geometry = True)

show_binary_header(data)
show_trace_header(data)

# csg_number = 100
# crg_number = 100
# cmp_number = 100
# cog_number = 12.5

# print(OFFt[:spread])

# index_CMP = np.where(CMPi == cmp_number)
# index_SRC = np.where(SRCi == csg_number)
# index_REC = np.where(RECi == crg_number)
# index_OFF = np.where(OFFt == cog_number)

# min_x = min(np.min(SPS[:,0]), np.min(RPS[:,0]))
# max_x = max(np.max(SPS[:,0]), np.max(RPS[:,0]))

# min_y = min(np.min(SPS[:,1]), np.min(RPS[:,1]))
# max_y = max(np.max(SPS[:,1]), np.max(RPS[:,1]))

# max_width = 10
# white_space = 100

# fig, ax = plt.subplots(ncols = 1, nrows = 5, figsize = (10, 9))

# ax[0].plot(rx, ry, "o")
# ax[0].plot(sx, sy, "o")

# ax[0].set_ylim([min_y - white_space, max_y + white_space])
# ax[0].set_xlim([min_x - white_space, max_x + white_space])

# ax[0].set_title("Entire acquisition geometry", fontsize = 15)
# ax[0].set_xlabel("UTM EAST [m]", fontsize = 12)
# ax[0].set_ylabel("UTM NORTH [m]", fontsize = 12)

# ax[1].plot(rx[index_SRC], ry[index_SRC], "o")
# ax[1].plot(sx[index_SRC], sy[index_SRC], "o")

# ax[1].set_ylim([min_y - white_space, max_y + white_space])
# ax[1].set_xlim([min_x - white_space, max_x + white_space])

# ax[1].set_title(f"Common shot gather number {csg_number}", fontsize = 15)
# ax[1].set_xlabel("UTM EAST [m]", fontsize = 12)
# ax[1].set_ylabel("UTM NORTH [m]", fontsize = 12)


# ax[2].plot(rx[index_REC], ry[index_REC], "o")
# ax[2].plot(sx[index_REC], sy[index_REC], "o")

# ax[2].set_ylim([min_y - white_space, max_y + white_space])
# ax[2].set_xlim([min_x - white_space, max_x + white_space])

# ax[2].set_title(f"Common receiver gather number {crg_number}", fontsize = 15)
# ax[2].set_xlabel("UTM EAST [m]", fontsize = 12)
# ax[2].set_ylabel("UTM NORTH [m]", fontsize = 12)

# ax[3].plot(rx[index_CMP], ry[index_CMP], "o")
# ax[3].plot(sx[index_CMP], sy[index_CMP], "o")
# ax[3].plot(CMPx[index_CMP], CMPy[index_CMP], "o", label = "CMPs")

# ax[3].set_ylim([min_y - white_space, max_y + white_space])
# ax[3].set_xlim([min_x - white_space, max_x + white_space])

# ax[3].set_title(f"Common mid point gather number {cmp_number}", fontsize = 15)
# ax[3].set_xlabel("UTM EAST [m]", fontsize = 12)
# ax[3].set_ylabel("UTM NORTH [m]", fontsize = 12)

# ax[4].plot(rx[index_OFF], ry[index_OFF], "o")
# ax[4].plot(sx[index_OFF], sy[index_OFF], "o")

# ax[4].set_ylim([min_y - white_space, max_y + white_space])
# ax[4].set_xlim([min_x - white_space, max_x + white_space])

# ax[4].set_title(f"Common offset gather number {cog_number} m", fontsize = 15)
# ax[4].set_xlabel("UTM EAST [m]", fontsize = 12)
# ax[4].set_ylabel("UTM NORTH [m]", fontsize = 12)

# fig.tight_layout()
# plt.show()

