import numpy as np
import segyio as sgy

'''
    Fixing original data: raw traces removed.
    Adding header properly: keywords updated.
'''

time_samples = 1501
time_spacing = 2e-3

rec_spacing = 25
src_spacing = 50

spread = 282

data = sgy.open("2D_Land_vibro_data_2ms/Line_001.sgy", ignore_geometry = True)
data = data.trace.raw[:].T

SPS = np.loadtxt("2D_Land_vibro_data_2ms/Line_001.SPS", skiprows = 20, usecols = [7,8,9], dtype = np.float32)
RPS = np.loadtxt("2D_Land_vibro_data_2ms/Line_001.RPS", skiprows = 20, usecols = [7,8,9], dtype = np.float32)

total_sources = len(SPS)
total_stations = len(RPS)

XPS = np.zeros((total_sources,2), dtype = int)

XPS[:,0] = np.arange(0, 2*total_sources, 2, dtype = int)
XPS[:,1] = np.arange(0, 2*total_sources, 2, dtype = int) + spread 

SRCi = np.zeros(spread*total_sources, dtype = int)
RECi = np.zeros(spread*total_sources, dtype = int)

CMPi = np.zeros(spread*total_sources, dtype = int)
CMPx = np.zeros(spread*total_sources, dtype = float)
CMPy = np.zeros(spread*total_sources, dtype = float)

OFFt = np.zeros(spread*total_sources, dtype = float)

sx = np.zeros(spread*total_sources, dtype = float)
sy = np.zeros(spread*total_sources, dtype = float)
sz = np.zeros(spread*total_sources, dtype = float)

rx = np.zeros(spread*total_sources, dtype = float)
ry = np.zeros(spread*total_sources, dtype = float)
rz = np.zeros(spread*total_sources, dtype = float)

tsl = np.arange(spread*total_sources, dtype = int)

seismic = np.zeros((time_samples, spread*total_sources), dtype = np.float32)

for i in range(total_sources):

    actives = slice(i*spread, i*spread + spread)

    sx[actives] = SPS[i,0]
    sy[actives] = SPS[i,1]
    sz[actives] = SPS[i,2]

    rx[actives] = RPS[XPS[i,0]:XPS[i,1],0]
    ry[actives] = RPS[XPS[i,0]:XPS[i,1],1]
    rz[actives] = RPS[XPS[i,0]:XPS[i,1],2]

    SRCi[actives] = i+1
    RECi[actives] = np.arange(spread, dtype = int) + 2*i + 1
    CMPi[actives] = np.arange(spread, dtype = int) + 4*i + 1

    CMPx[actives] = sx[actives] - 0.5*(sx[actives] - rx[actives])    
    CMPy[actives] = sy[actives] - 0.5*(sy[actives] - ry[actives])

    OFFt[actives] = np.arange(spread)*rec_spacing - 0.5*(spread-1)*rec_spacing   

    seismic[:,i*spread:i*spread+spread] = data[:,i*(spread+2)+2:i*(spread+2)+spread+2]

#----------------------------------------------------------------------------------------------------

sgy.tools.from_array2D("2D_Land_vibro_data_2ms/seismic_raw.sgy", seismic.T)

data = sgy.open("2D_Land_vibro_data_2ms/seismic_raw.sgy", "r+", ignore_geometry = True)

data.bin[sgy.BinField.JobID]                 = 1
data.bin[sgy.BinField.LineNumber]            = 1
data.bin[sgy.BinField.ReelNumber]            = 1
data.bin[sgy.BinField.Interval]              = int(time_spacing*1e6)
data.bin[sgy.BinField.IntervalOriginal]      = int(time_spacing*1e6)
data.bin[sgy.BinField.Samples]               = time_samples
data.bin[sgy.BinField.SamplesOriginal]       = time_samples
data.bin[sgy.BinField.Format]                = 1
data.bin[sgy.BinField.SortingCode]           = 1
data.bin[sgy.BinField.MeasurementSystem]     = 1
data.bin[sgy.BinField.ImpulseSignalPolarity] = 1


for idx, key in enumerate(data.header):

    key.update({sgy.TraceField.TRACE_SEQUENCE_LINE                    : int(tsl[idx])              })
    key.update({sgy.TraceField.TRACE_SEQUENCE_FILE                    : int(SRCi[idx])              })
    key.update({sgy.TraceField.FieldRecord                            : int(SRCi[idx])             })
    key.update({sgy.TraceField.TraceNumber                            : int(RECi[idx])             })
    key.update({sgy.TraceField.EnergySourcePoint                      : 0                     })
    key.update({sgy.TraceField.CDP                                    : int(CMPi[idx])             })
    key.update({sgy.TraceField.CDP_TRACE                              : int(CMPi[idx])             })
    key.update({sgy.TraceField.TraceIdentificationCode                : int(tsl[idx])              })
    key.update({sgy.TraceField.NSummedTraces                          : 0                     })
    key.update({sgy.TraceField.NStackedTraces                         : 0                     })
    key.update({sgy.TraceField.DataUse                                : 0                     })
    key.update({sgy.TraceField.offset                                 : int(OFFt[idx]*100)             })
    key.update({sgy.TraceField.ReceiverGroupElevation                 : int(rz[idx]*100)      })
    key.update({sgy.TraceField.SourceSurfaceElevation                 : int(sz[idx]*100)      })
    key.update({sgy.TraceField.SourceDepth                            : 0                     })
    key.update({sgy.TraceField.ReceiverDatumElevation                 : 0                     })
    key.update({sgy.TraceField.SourceDatumElevation                   : 0                     })
    key.update({sgy.TraceField.SourceWaterDepth                       : 0                     })
    key.update({sgy.TraceField.ElevationScalar                        : 100                   })
    key.update({sgy.TraceField.SourceGroupScalar                      : 100                   })
    key.update({sgy.TraceField.SourceX                                : int(sx[idx]*100)      })
    key.update({sgy.TraceField.SourceY                                : int(sy[idx]*100)      })
    key.update({sgy.TraceField.GroupX                                 : int(rx[idx]*100)      })
    key.update({sgy.TraceField.GroupY                                 : int(ry[idx]*100)      })
    key.update({sgy.TraceField.CoordinateUnits                        : 1                     })
    key.update({sgy.TraceField.WeatheringVelocity                     : 0                     })
    key.update({sgy.TraceField.SubWeatheringVelocity                  : 0                     })
    key.update({sgy.TraceField.SourceUpholeTime                       : 0                     })
    key.update({sgy.TraceField.GroupUpholeTime                        : 0                     })
    key.update({sgy.TraceField.SourceStaticCorrection                 : 0                     })
    key.update({sgy.TraceField.GroupStaticCorrection                  : 0                     })
    key.update({sgy.TraceField.TotalStaticApplied                     : 0                     })
    key.update({sgy.TraceField.LagTimeA                               : 0                     })
    key.update({sgy.TraceField.LagTimeB                               : 0                     })
    key.update({sgy.TraceField.DelayRecordingTime                     : 0                     })
    key.update({sgy.TraceField.MuteTimeStart                          : 0                     })
    key.update({sgy.TraceField.MuteTimeEND                            : 0                     })
    key.update({sgy.TraceField.TRACE_SAMPLE_COUNT                     : time_samples          })
    key.update({sgy.TraceField.TRACE_SAMPLE_INTERVAL                  : int(time_spacing*1e6) })
    key.update({sgy.TraceField.GainType                               : 1                     })
    key.update({sgy.TraceField.InstrumentGainConstant                 : 0                     })
    key.update({sgy.TraceField.InstrumentInitialGain                  : 0                     })
    key.update({sgy.TraceField.Correlated                             : 0                     })
    key.update({sgy.TraceField.SweepFrequencyStart                    : 0                     })
    key.update({sgy.TraceField.SweepFrequencyEnd                      : 0                     })
    key.update({sgy.TraceField.SweepLength                            : 0                     })
    key.update({sgy.TraceField.SweepType                              : 0                     })
    key.update({sgy.TraceField.SweepTraceTaperLengthStart             : 0                     })
    key.update({sgy.TraceField.SweepTraceTaperLengthEnd               : 0                     })
    key.update({sgy.TraceField.TaperType                              : 0                     })
    key.update({sgy.TraceField.AliasFilterFrequency                   : 0                     })
    key.update({sgy.TraceField.AliasFilterSlope                       : 0                     })
    key.update({sgy.TraceField.NotchFilterFrequency                   : 0                     })
    key.update({sgy.TraceField.NotchFilterSlope                       : 0                     })
    key.update({sgy.TraceField.LowCutFrequency                        : 0                     })
    key.update({sgy.TraceField.HighCutFrequency                       : 0                     })
    key.update({sgy.TraceField.LowCutSlope                            : 0                     })
    key.update({sgy.TraceField.HighCutSlope                           : 0                     })
    key.update({sgy.TraceField.YearDataRecorded                       : 0                     })
    key.update({sgy.TraceField.DayOfYear                              : 0                     })
    key.update({sgy.TraceField.HourOfDay                              : 0                     })
    key.update({sgy.TraceField.MinuteOfHour                           : 0                     })
    key.update({sgy.TraceField.SecondOfMinute                         : 0                     })
    key.update({sgy.TraceField.TimeBaseCode                           : 1                     })
    key.update({sgy.TraceField.TraceWeightingFactor                   : 0                     })
    key.update({sgy.TraceField.GeophoneGroupNumberRoll1               : 0                     })
    key.update({sgy.TraceField.GeophoneGroupNumberFirstTraceOrigField : 0                     })
    key.update({sgy.TraceField.GeophoneGroupNumberLastTraceOrigField  : 0                     })
    key.update({sgy.TraceField.GapSize                                : 0                     })
    key.update({sgy.TraceField.OverTravel                             : 0                     })
    key.update({sgy.TraceField.CDP_X                                  : int(CMPx[idx]*100)    })
    key.update({sgy.TraceField.CDP_Y                                  : int(CMPy[idx]*100)    })
    key.update({sgy.TraceField.INLINE_3D                              : 0                     })
    key.update({sgy.TraceField.CROSSLINE_3D                           : 0                     })
    key.update({sgy.TraceField.ShotPoint                              : 0                     })
    key.update({sgy.TraceField.ShotPointScalar                        : 0                     })
    key.update({sgy.TraceField.TraceValueMeasurementUnit              : 0                     })
    key.update({sgy.TraceField.TransductionConstantMantissa           : 0                     })
    key.update({sgy.TraceField.TransductionConstantPower              : 0                     })
    key.update({sgy.TraceField.TraceIdentifier                        : 0                     })
    key.update({sgy.TraceField.ScalarTraceHeader                      : 0                     })
    key.update({sgy.TraceField.SourceType                             : 0                     })
    key.update({sgy.TraceField.SourceEnergyDirectionMantissa          : 0                     })
    key.update({sgy.TraceField.SourceEnergyDirectionExponent          : 0                     })
    key.update({sgy.TraceField.SourceMeasurementUnit                  : 0                     })
    key.update({sgy.TraceField.UnassignedInt1                         : 0                     })
    key.update({sgy.TraceField.UnassignedInt2                         : 0                     })
    
data.close()

