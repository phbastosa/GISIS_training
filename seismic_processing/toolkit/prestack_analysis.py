import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

class Prestack:
    def __init__(self):
        pass

    def import_segy_data(self, data_path):
        self.data = sgy.open(data_path, ignore_geometry = True)
        self.seismic = self.data.trace.raw[:].T        
        
    def import_binary_data(self, time_samples, trace_number, data_path): 
        aux = np.fromfile(data_path, dtype = np.float32, count = time_samples*trace_number)
        self.seismic = np.reshape(aux, [time_samples, trace_number], order = "F") 

    def export_binary_data(self, data_path):
        self.seismic.flatten("F").astype(np.float32, order = "F").tofile(data_path)

    def import_geometry(self, source_path, receiver_path, relational_path):
        self.sources = np.loadtxt(source_path, dtype = np.float32, delimiter = ",", comments = "#")
        self.receivers = np.loadtxt(receiver_path, dtype = np.float32, delimiter = ",", comments = "#")
        self.relational = np.loadtxt(relational_path, dtype = np.int32, delimiter = ",", comments = "#")

    def add_binary_data_trace_header(self, time_spacing, gather_index, trace_number, output_path):

        nt = len(self.seismic)

        gather_index -= 1

        gather_slice = np.arange(gather_index*trace_number, gather_index*trace_number + trace_number, dtype = int)    

        offset = np.sqrt((self.receivers[gather_slice, 0] - self.sources[gather_index, 0])**2 + 
                         (self.receivers[gather_slice, 1] - self.sources[gather_index, 1])**2 + 
                         (self.receivers[gather_slice, 2] - self.sources[gather_index, 2])**2)  

        cmpx = 0.5 * (self.receivers[gather_slice, 0] + self.sources[gather_index, 0]) 
        cmpy = 0.5 * (self.receivers[gather_slice, 1] + self.sources[gather_index, 1]) 

        cmp = np.sqrt(cmpx*cmpx + cmpy*cmpy)
        cmpt = gather_index*np.linspace(1, trace_number, trace_number, dtype = int) 

        fldr = 1001 + gather_index*trace_number*np.ones(trace_number, dtype = int)

        sgy.tools.from_array2D(output_path, self.seismic.T)

        segy = sgy.open(output_path, "r+", ignore_geometry = True)

        segy.bin[sgy.BinField.Interval]              = int(time_spacing*1e6)
        segy.bin[sgy.BinField.IntervalOriginal]      = int(time_spacing*1e6)
        segy.bin[sgy.BinField.Format]                = 1
        segy.bin[sgy.BinField.SortingCode]           = 1
        segy.bin[sgy.BinField.MeasurementSystem]     = 1
        segy.bin[sgy.BinField.ImpulseSignalPolarity] = 1

        tracl = 1 + gather_index*trace_number*np.arange(trace_number, dtype = int) 

        for idx, key in enumerate(segy.header):

            key.update({sgy.TraceField.TRACE_SEQUENCE_LINE     : int(tracl[idx])                          })
            key.update({sgy.TraceField.TRACE_SEQUENCE_FILE     : int(fldr[idx])                           })
            key.update({sgy.TraceField.TRACE_SAMPLE_INTERVAL   : int(time_spacing*1e6)                    })
            key.update({sgy.TraceField.TRACE_SAMPLE_COUNT      : nt                                       })
            key.update({sgy.TraceField.FieldRecord             : int(fldr[idx])                           })
            key.update({sgy.TraceField.TraceNumber             : int(tracl[idx])                          })
            key.update({sgy.TraceField.CDP                     : int(cmpt[idx])                            })
            key.update({sgy.TraceField.CDP_TRACE               : int(cmp[idx])                           })
            key.update({sgy.TraceField.TraceIdentificationCode : int(tracl[idx])                          })
            key.update({sgy.TraceField.offset                  : int(offset[idx])                         })
            key.update({sgy.TraceField.ReceiverGroupElevation  : int(self.receivers[gather_slice[idx],2]) })
            key.update({sgy.TraceField.SourceSurfaceElevation  : int(self.sources[gather_index,2])        })
            key.update({sgy.TraceField.ElevationScalar         : 100                                      })
            key.update({sgy.TraceField.SourceGroupScalar       : 100                                      })
            key.update({sgy.TraceField.SourceX                 : int(self.sources[gather_index,0])        })
            key.update({sgy.TraceField.SourceY                 : int(self.sources[gather_index,1])        })
            key.update({sgy.TraceField.SourceDepth             : int(self.sources[gather_index,2])        })
            key.update({sgy.TraceField.GroupX                  : int(self.receivers[gather_slice[idx],0]) })
            key.update({sgy.TraceField.GroupY                  : int(self.receivers[gather_slice[idx],1]) })
            key.update({sgy.TraceField.GroupWaterDepth         : int(self.receivers[gather_slice[idx],2]) })
            key.update({sgy.TraceField.CoordinateUnits         : 1                                        })
            key.update({sgy.TraceField.GainType                : 1                                        })
            key.update({sgy.TraceField.TimeBaseCode            : 1                                        })
            key.update({sgy.TraceField.CDP_X                   : int(cmpx[idx])                           })
            key.update({sgy.TraceField.CDP_Y                   : int(cmpy[idx])                           })
        
        segy.close()

    def show_binary_header(self):
        binHeader = sgy.binfield.keys
        print("\n Checking binary header \n")
        print(f"{'key': >25s} {'byte': ^6s} {'value': ^7s} \n")
        for k, v in binHeader.items():
            if v in self.data.bin:
                print(f"{k: >25s} {str(v): ^6s} {str(self.data.bin[v]): ^7s}")

    def show_trace_header(self):
        traceHeader = sgy.tracefield.keys
        print("\n Checking trace header \n")
        print(f"{'Trace header': >40s} {'byte': ^6s} {'first': ^11s} {'last': ^11s} \n")
        for k, v in traceHeader.items():
            first = self.data.attributes(v)[0][0]
            last = self.data.attributes(v)[self.data.tracecount-1][0]
            print(f"{k: >40s} {str(v): ^6s} {str(first): ^11s} {str(last): ^11s}")

    def plot_geometry(self):
        fig, ax = plt.subplots(num = "Geometry Plot", figsize = (10,7))

        ax.plot(1e-2*self.receivers[:,0], 1e-2*self.receivers[:,1], "og", label = "Receivers")
        ax.plot(1e-2*self.sources[:,0], 1e-2*self.sources[:,1], "ok", label = "Sources")
        
        ax.set_xlabel("UTM E [m]", fontsize = 15)
        ax.set_ylabel("UTM N [m]", fontsize = 15)
        
        ax.legend(loc = "upper left", fontsize = 12)

        fig.tight_layout()
        plt.show()
