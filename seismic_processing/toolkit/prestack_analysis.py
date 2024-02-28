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

        s2us = 1e6
        geom_scale = 100

        nt = len(self.seismic)
        dt = int(time_spacing*s2us)

        rec_beg = self.relational[gather_index - 1, 0]    
        rec_end = self.relational[gather_index - 1, 1]    

        offset = np.sqrt((self.receivers[rec_beg:rec_end, 0] - self.sources[gather_index-1, 0])**2 + 
                         (self.receivers[rec_beg:rec_end, 1] - self.sources[gather_index-1, 1])**2 + 
                         (self.receivers[rec_beg:rec_end, 2] - self.sources[gather_index-1, 2])**2)  

        offset *= geom_scale

        cmpx = 0.5 * (self.receivers[rec_beg:rec_end, 0] + self.sources[gather_index-1, 0]) 
        cmpy = 0.5 * (self.receivers[rec_beg:rec_end, 1] + self.sources[gather_index-1, 1]) 

        cmp = np.sqrt(cmpx*cmpx + cmpy*cmpy) * geom_scale
        cmpt = gather_index + np.arange(trace_number, dtype = int) 

        tsf = 1000 + gather_index 
        tsl = 1 + np.arange((gather_index-1)*trace_number, (gather_index-1)*trace_number + trace_number, dtype = int)
        fldr = 1 + np.arange(trace_number, dtype = int)

        sx = int(self.sources[gather_index,0]*geom_scale)    
        sy = int(self.sources[gather_index,1]*geom_scale)    
        sz = int(self.sources[gather_index,2]*geom_scale)    

        gx = np.array(self.receivers[rec_beg:rec_end, 0]*geom_scale, dtype = int)    
        gy = np.array(self.receivers[rec_beg:rec_end, 1]*geom_scale, dtype = int)    
        gz = np.array(self.receivers[rec_beg:rec_end, 2]*geom_scale, dtype = int)    

        sgy.tools.from_array2D(output_path, self.seismic.T)

        self.data = sgy.open(output_path, "r+", ignore_geometry = True)

        self.data.bin[sgy.BinField.JobID]                 = gather_index
        self.data.bin[sgy.BinField.Interval]              = dt
        self.data.bin[sgy.BinField.IntervalOriginal]      = dt
        self.data.bin[sgy.BinField.Format]                = 1
        self.data.bin[sgy.BinField.SortingCode]           = 1
        self.data.bin[sgy.BinField.MeasurementSystem]     = 1
        self.data.bin[sgy.BinField.ImpulseSignalPolarity] = 1

        for idx, key in enumerate(self.data.header):
        
            key.update({sgy.TraceField.TRACE_SEQUENCE_LINE                    : tsl[idx]         })
            key.update({sgy.TraceField.TRACE_SEQUENCE_FILE                    : tsf              })
            key.update({sgy.TraceField.FieldRecord                            : tsl[idx]         })
            key.update({sgy.TraceField.TraceNumber                            : fldr[idx]        })
            key.update({sgy.TraceField.EnergySourcePoint                      : 0                })
            key.update({sgy.TraceField.CDP                                    : int(cmp[idx])    })
            key.update({sgy.TraceField.CDP_TRACE                              : int(cmpt[idx])   })
            key.update({sgy.TraceField.TraceIdentificationCode                : tsl[idx]         })
            key.update({sgy.TraceField.NSummedTraces                          : 0                })
            key.update({sgy.TraceField.NStackedTraces                         : 0                })
            key.update({sgy.TraceField.DataUse                                : 0                })
            key.update({sgy.TraceField.offset                                 : int(offset[idx]) })
            key.update({sgy.TraceField.ReceiverGroupElevation                 : gz[idx]          })
            key.update({sgy.TraceField.SourceSurfaceElevation                 : sz               })
            key.update({sgy.TraceField.SourceDepth                            : 0                })
            key.update({sgy.TraceField.ReceiverDatumElevation                 : 0                })
            key.update({sgy.TraceField.SourceDatumElevation                   : 0                })
            key.update({sgy.TraceField.SourceWaterDepth                       : 0                })
            key.update({sgy.TraceField.ElevationScalar                        : geom_scale       })
            key.update({sgy.TraceField.SourceGroupScalar                      : geom_scale       })
            key.update({sgy.TraceField.SourceX                                : sx               })
            key.update({sgy.TraceField.SourceY                                : sy               })
            key.update({sgy.TraceField.GroupX                                 : gx[idx]          })
            key.update({sgy.TraceField.GroupY                                 : gy[idx]          })
            key.update({sgy.TraceField.CoordinateUnits                        : 1                })
            key.update({sgy.TraceField.WeatheringVelocity                     : 0                })
            key.update({sgy.TraceField.SubWeatheringVelocity                  : 0                })
            key.update({sgy.TraceField.SourceUpholeTime                       : 0                })
            key.update({sgy.TraceField.GroupUpholeTime                        : 0                })
            key.update({sgy.TraceField.SourceStaticCorrection                 : 0                })
            key.update({sgy.TraceField.GroupStaticCorrection                  : 0                })
            key.update({sgy.TraceField.TotalStaticApplied                     : 0                })
            key.update({sgy.TraceField.LagTimeA                               : 0                })
            key.update({sgy.TraceField.LagTimeB                               : 0                })
            key.update({sgy.TraceField.DelayRecordingTime                     : 0                })
            key.update({sgy.TraceField.MuteTimeStart                          : 0                })
            key.update({sgy.TraceField.MuteTimeEND                            : 0                })
            key.update({sgy.TraceField.TRACE_SAMPLE_COUNT                     : nt               })
            key.update({sgy.TraceField.TRACE_SAMPLE_INTERVAL                  : dt               })
            key.update({sgy.TraceField.GainType                               : 1                })
            key.update({sgy.TraceField.InstrumentGainConstant                 : 0                })
            key.update({sgy.TraceField.InstrumentInitialGain                  : 0                })
            key.update({sgy.TraceField.Correlated                             : 0                })
            key.update({sgy.TraceField.SweepFrequencyStart                    : 0                })
            key.update({sgy.TraceField.SweepFrequencyEnd                      : 0                })
            key.update({sgy.TraceField.SweepLength                            : 0                })
            key.update({sgy.TraceField.SweepType                              : 0                })
            key.update({sgy.TraceField.SweepTraceTaperLengthStart             : 0                })
            key.update({sgy.TraceField.SweepTraceTaperLengthEnd               : 0                })
            key.update({sgy.TraceField.TaperType                              : 0                })
            key.update({sgy.TraceField.AliasFilterFrequency                   : 0                })
            key.update({sgy.TraceField.AliasFilterSlope                       : 0                })
            key.update({sgy.TraceField.NotchFilterFrequency                   : 0                })
            key.update({sgy.TraceField.NotchFilterSlope                       : 0                })
            key.update({sgy.TraceField.LowCutFrequency                        : 0                })
            key.update({sgy.TraceField.HighCutFrequency                       : 0                })
            key.update({sgy.TraceField.LowCutSlope                            : 0                })
            key.update({sgy.TraceField.HighCutSlope                           : 0                })
            key.update({sgy.TraceField.YearDataRecorded                       : 0                })
            key.update({sgy.TraceField.DayOfYear                              : 0                })
            key.update({sgy.TraceField.HourOfDay                              : 0                })
            key.update({sgy.TraceField.MinuteOfHour                           : 0                })
            key.update({sgy.TraceField.SecondOfMinute                         : 0                })
            key.update({sgy.TraceField.TimeBaseCode                           : 1                })
            key.update({sgy.TraceField.TraceWeightingFactor                   : 0                })
            key.update({sgy.TraceField.GeophoneGroupNumberRoll1               : 0                })
            key.update({sgy.TraceField.GeophoneGroupNumberFirstTraceOrigField : 0                })
            key.update({sgy.TraceField.GeophoneGroupNumberLastTraceOrigField  : 0                })
            key.update({sgy.TraceField.GapSize                                : 0                })
            key.update({sgy.TraceField.OverTravel                             : 0                })
            key.update({sgy.TraceField.CDP_X                                  : int(cmpx[idx])   })
            key.update({sgy.TraceField.CDP_Y                                  : int(cmpy[idx])   })
            key.update({sgy.TraceField.INLINE_3D                              : 0                })
            key.update({sgy.TraceField.CROSSLINE_3D                           : 0                })
            key.update({sgy.TraceField.ShotPoint                              : 0                })
            key.update({sgy.TraceField.ShotPointScalar                        : 0                })
            key.update({sgy.TraceField.TraceValueMeasurementUnit              : 0                })
            key.update({sgy.TraceField.TransductionConstantMantissa           : 0                })
            key.update({sgy.TraceField.TransductionConstantPower              : 0                })
            key.update({sgy.TraceField.TraceIdentifier                        : 0                })
            key.update({sgy.TraceField.ScalarTraceHeader                      : 0                })
            key.update({sgy.TraceField.SourceType                             : 0                })
            key.update({sgy.TraceField.SourceEnergyDirectionMantissa          : 0                })
            key.update({sgy.TraceField.SourceEnergyDirectionExponent          : 0                })
            key.update({sgy.TraceField.SourceMeasurementUnit                  : 0                })
            key.update({sgy.TraceField.UnassignedInt1                         : 0                })
            key.update({sgy.TraceField.UnassignedInt2                         : 0                })
        
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
