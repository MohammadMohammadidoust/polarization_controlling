import pyvisa
import matplotlib.pyplot as plt
import time
from quantiphy import Quantity
import numpy as np

address = "USB0::0x5345::0x1235::2052231::INSTR"
resource = pyvisa.ResourceManager()
instrument = resource.open_resource(address)

def getSampleRate(outputs, bits, samplingPoints, timeBase):
    # see programming manual page 58
    maxSampleRates = {'single':{'8':1E9, '12': 500E6, '14':125E6}, 
                     'dual'  :{'8':1E9, '12': 500E6, '14':125E6},
                     'quad'  :{'8':1E9, '12': 500E6, '14':125E6}}
    maxRate = maxSampleRates[outputs][bits]
    samplingPointsPerDiv = {'1k':50, '10k':500, '100k':5E3,'1M':50E3,'10M':500E3,'25M':1.25E6,'50M':2.5E6,'100M':5E6,'250M':12.5E6}
    samplePts = samplingPointsPerDiv[samplingPoints]
    if maxRate > samplePts / timeBase:
        return samplePts / timeBase
    else:
        return maxRate

resp = instrument.query("*IDN?")
print(resp)

# initial parameters
print("*********************\n\n*****initial values\n\n*******************")
print('horizontal scale: ', instrument.query(":HORIZONTAL:SCALE?"))
print('horizontal offset: ', instrument.query(":HORIZONTAL:OFFSET?"))
print('ch1 scale: ', instrument.query(":CH1:SCALE?"))
print('ch1 voltage offset: ', instrument.query(":CH1:OFFSET?"))
print('ch1 coupling: ', instrument.query(":CH1:COUPLING?"))
print('trigger source: ', instrument.query(":TRIGGER:SINGLE:EDGE:SOURCE?"))
print('trigger level: ', instrument.query(":TRIGGER:SINGLE:EDGE:LEVEL ?"))
print('trigger coupling: ', instrument.query(":TRIGGER:SINGLE:EDGE:COUPLING?"))
print('ch1 display: ', instrument.query(":CH1:DISP?"))
print('acquire mode: ', instrument.query(":ACQUIRE:MODE?"))
print('dep memory: ', instrument.query(":ACQUIRE:DEPMEM?"))
print('precision: ', instrument.query(":ACQUIRE:PRECISION?"))
print('waveform range: ', instrument.query(":WAVEFORM:RANGE?"))
print('ch2 display: ', instrument.query(":CH2:DISP?"))


#instrument.write(":AUTOSET")
#print("progress: ", instrument.query(":AUTOSET:PROGRESS?"))
#time.sleep(30)
#print("progress: ", instrument.query(":AUTOSET:PROGRESS?"))

####testing parameters:
instrument.write(":HORIZONTAL:SCALE 2.0ms")

instrument.write(":MEAS:DISP ON")
instrument.write(":MEASure:SOURce CH1")
print('measure source: ', instrument.query(":MEASure:SOURce?"))
ch1_offset = np.float64(instrument.query(":MEASure:VBASE?"))
print('measure VBase channel 1: ', ch1_offset)

instrument.write(":MEASure:SOURce CH2")
print('measure source: ', instrument.query(":MEASure:SOURce?"))
ch2_offset = np.float64(instrument.query(":MEASure:VBASE?"))
print('measure VBase channel 2: ', ch2_offset)

instrument.write(":MEASure:SOURce CH3")
print('measure source: ', instrument.query(":MEASure:SOURce?"))
ch3_offset = np.float64(instrument.query(":MEASure:VBASE?"))
print('measure VBase channel 3: ', ch3_offset)

#instrument.write(":MEASure:SOURce CH4")
#print('measure source: ', instrument.query(":MEASure:SOURce?"))
#ch4_offset = np.float64(instrument.query(":MEASure:VBASE?"))
#print('measure VBase channel 4: ', ch4_offset)

instrument.write(":MEAS:DISP OFF")

off1 = ch1_offset / 0.01           #0.01 is 10mv means the channel voltage scale
off2 = ch2_offset / 0.01
off3 = ch3_offset / 0.01
#off4 = ch4_offset / 0.01

#print('measure CH2 scale: ', off2)

instrument.write(":CH1:OFFSET 0V")
instrument.write(":CH1:SCALE 10mv")
instrument.write(":CH1:COUPLING DC")

#instrument.write(":ACQUIRE:MODE SAMPLE")
#instrument.write(":ACQUIRE:DEPMEM 1K")
#a = instrument.write(":ACQ:PREC 8")
#print("aaaaaaaaaaaaaaaaaa: ", a)
#print("aaa prec", instrument.query(":ACQ:PREC?") )

instrument.write(":CH2:DISP ON")
instrument.write(":CH2:OFFSET 0V")
instrument.write(":CH2:SCALE 10mv")

instrument.write(":CH3:DISP ON")
instrument.write(":CH3:OFFSET 0V")
instrument.write(":CH3:SCALE 10mv")

#instrument.write(":CH4:DISP ON")
#instrument.write(":CH4:OFFSET 0V")
#instrument.write(":CH4:SCALE 10mv")

time.sleep(5)

auto_volt_wave1 = []
volt_wave1 = []
volt_wave2 = []
volt_wave3 = []
volt_wave4 = []

time_list = []
instrument.write(":WAV:BEG CH1")
instrument.write(":WAVEFORM:RANGE 0,1000")
adc_wave = instrument.query_binary_values(":WAV:FETC?", datatype = 'h')
instrument.write(":WAV:BEG CH2")
instrument.write(":WAVEFORM:RANGE 0,1000")
adc_wave2 = instrument.query_binary_values(":WAV:FETC?", datatype = 'h')
instrument.write(":WAV:BEG CH3")
instrument.write(":WAVEFORM:RANGE 0,1000")
adc_wave3 = instrument.query_binary_values(":WAV:FETC?", datatype = 'h')
#instrument.write(":WAV:BEG CH4")
#instrument.write(":WAVEFORM:RANGE 0,1000")
#adc_wave4 = instrument.query_binary_values(":WAV:FETC?", datatype = 'h')
instrument.write(":WAV:END")

scale_time = Quantity(instrument.query(":HORI:SCAL?")).real
print("scale time: ", scale_time)
scale_voltage = Quantity(instrument.query(":CH1:SCAL?")).real
offset_divisions = float(instrument.query(":CH1:OFFS?"))

#for index, value in enumerate(adc_wave):
#    volt_wave1.append((float(value) / 6400 - offset_divisions) * scale_voltage)
#    time_list.append(scale_time * index)
sample_rate = getSampleRate('quad', '8', '1k', 2E-3)
print("sample rate: ", sample_rate)
for index in range(len(adc_wave)):
    #auto_volt_wave1.append((float(adc_wave[index]) / 6400 - offset_divisions) * scale_voltage)
    #volt_wave2.append((float(adc_wave2[index]) / 6400 - offset_divisions) * scale_voltage)
    #volt_wave3.append((float(adc_wave2[index]) / 6400 - offset_divisions) * scale_voltage)
    #volt_wave4.append((float(adc_wave2[index]) / 6400 - offset_divisions) * scale_voltage)
    volt_wave1.append((float(adc_wave[index]) / 6400 - off1) * scale_voltage)
    volt_wave2.append((float(adc_wave2[index]) / 6400 - off2) * scale_voltage)
    volt_wave3.append((float(adc_wave3[index]) / 6400 - off3) * scale_voltage)
    #volt_wave4.append((float(adc_wave4[index]) / 6400 - off4) * scale_voltage)
    #time_list.append(scale_time * index)
    time_list.append(index/sample_rate)


print("len volt: ", len(volt_wave1))
print("len time: ", len(time_list))



print('new ch1 voltage offset: ', instrument.query(":CH1:OFFSET?"))
print('new ch1 scale: ', instrument.query(":CH1:SCALE?"))
print('new horizontal scale: ', instrument.query(":HORIZONTAL:SCALE?"))
print('new acquire mode: ', instrument.query(":ACQUIRE:MODE?"))
print('new dep memory: ', instrument.query(":ACQUIRE:DEPMEM?"))
print('new precision: ', instrument.query(":ACQUIRE:PRECISION?"))
print('new ch2 voltage offset: ', instrument.query(":CH2:OFFSET?"))
print('new ch2 scale: ', instrument.query(":CH2:SCALE?"))
print('new ch1 display: ', instrument.query(":CH1:DISP?"))
print('new ch2 display: ', instrument.query(":CH2:DISP?"))
'''
print("*********************\n\n*****final values\n\n*******************")
print('horizontal scale: ', instrument.query(":HORIZONTAL:SCALE?"))
print('horizontal offset: ', instrument.query(":HORIZONTAL:OFFSET?"))
print('ch1 scale: ', instrument.query(":CH1:SCALE?"))
print('ch1 voltage offset: ', instrument.query(":CH1:OFFSET?"))
print('ch1 coupling: ', instrument.query(":CH1:COUPLING?"))
print('trigger source: ', instrument.query(":TRIGGER:SINGLE:EDGE:SOURCE?"))
print('trigger level: ', instrument.query(":TRIGGER:SINGLE:EDGE:"))
print('trigger coupling: ', instrument.query(":TRIGGER:SINGLE:EDGE:COUPLING?"))
print('ch1 display: ', instrument.query(":CH1:DISP?"))
print('acquire mode: ', instrument.query(":ACQUIRE:MODE?"))
print('dep memory: ', instrument.query(":ACQUIRE:DEPMEM?"))
print('precision: ', instrument.query(":ACQUIRE:PRECISION?"))
print('waveform range: ', instrument.query(":WAVEFORM:RANGE?"))
'''


# Use MatPlotLib to plot the waveforms 
fig, ax = plt.subplots()

ax.set(xlabel='time (S)', ylabel='voltage (V)', title='WAVEFORM')

ax.plot(time_list, volt_wave1, "-r", label="CH1")
#ax.plot(time_list, auto_volt_wave1, "-c", label="CH1_Auto")
ax.plot(time_list, volt_wave2, "-g", label="CH2")
ax.plot(time_list,volt_wave3, "-b", label="CH3")
#ax.plot(time_list, volt_wave4, "-c", label="CH4")

ax.grid()
plt.legend(loc="upper left")
plt.xlim([0, time_list[-1]])
# verticalHeight = VOLT_MULT[VOLTS_PER_DIVISION] * VOLT_DIVISIONS / 2
# plt.ylim([-verticalHeight, verticalHeight])
plt.show()

print("COMPLETED!")
