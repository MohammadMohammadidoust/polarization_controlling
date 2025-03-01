import pyvisa
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantiphy import Quantity
from scipy.signal import find_peaks

class OWON:
    def __init__(self,conf_dict):
        self.configs = conf_dict
        self.hv_qber = None
        self.pm_qber = None
        self.qber = self.hv_qber
        self.resource = pyvisa.ResourceManager()
        self.resource_address = self.configs['scope']['owon']['resource_address']
        self.instrument = self.resource.open_resource(self.resource_address)
        self.channels_dict = self.configs['scope']['owon']['channels']
        self.channels = list(self.configs['scope']['owon']['channels'].values())
        self.wave_amplitude = self.configs['scope']['owon']['wave']['amplitude']
        self.wave_frequency = self.configs['scope']['owon']['wave']['frequency']
        self.wave_period = 1.0/self.wave_frequency
        self.pduty1 = self.configs['scope']['owon']['wave']['pduty1']
        self.pduty2 = self.configs['scope']['owon']['wave']['pduty2']
        self.nduty1 = self.configs['scope']['owon']['wave']['nduty1']
        self.nduty2 = self.configs['scope']['owon']['wave']['nduty2']
        self.v_scale = self.configs['scope']['owon']['wave']['voltage_scale']  #must be in V unit to adjust offset of channels
        self.channels_params = {1: {"vbase": None, "offset": None},
                                2: {"vbase": None, "offset": None},
                                3: {"vbase": None, "offset": None},
                                4: {"vbase": None, "offset": None}}
        self.result_dict = {"Unix_time": [], "hv_qber": [], "pm_qber": [],
                            "qber": [], "Additional_data": []}

    def send(self, command):
        self.instrument.write(str(command))

    def query(self, command):
        response = self.instrument.query(str(command)).strip()
        return response

    def auto_set_device(self):
        self.send(self.configs['scope']['owon']['commands']['autoset'])
        print("###########start autoset device")
        print("###########be patient")
        print("Progress: ", self.query(
            self.configs['scope']['owon']['queries']['autoset_progress']))
        time.sleep(30)
        progress = int(self.query(
            self.configs['scope']['owon']['queries']['autoset_progress']))
        print("progress: ", progress)
        while progress < 100 :
            print("still needs patience!")
            time.sleep(10)
        print("done!")

    def measure_params(self):
        self.send(
            self.configs['scope']['owon']['commands']['measure_params']['turn_on'])
        for channel in self.channels:
            source_command = self.configs['scope']['owon']['commands']['measure_params']['source']
            self.send(source_command.format(channel))
            self.channels_params[channel]['vbase'] = np.float64(
                self.query(self.configs['scope']['owon']['commands']['measure_params']['vbase']))
            self.channels_params[channel]['offset'] = self.channels_params[channel]['vbase'] / self.v_scale
        self.send(self.configs['scope']['owon']['commands']['measure_params']['turn_off'])

    def initialise(self):
        for command in self.configs['scope']['owon']['commands']['initialise']['general']:
            self.send(command)
        for channel in self.channels:
            for command in self.configs['scope']['owon']['commands']['initialise']['channel']:
                self.send(command.format(channel))
        self.measure_params()

        time.sleep(10)
        print("initialise done!")

    def print_device_parameters(self):
        for general_query in self.configs['scope']['owon']['queries']['parameters']['general']:
                print(general_query, self.query(
                    self.configs['scope']['owon']['queries']['parameters']['general'][general_query]))
        for channel in self.channels:
            for channel_query in self.configs['scope']['owon']['queries']['parameters']['channel']:
                print(channel_query, self.query(
                    self.configs['scope']['owon']['queries']['parameters']['channel'][channel_query].format(channel)))

    def get_sample_rate(self, outputs):
        # see programming manual page 58
        self.update_device_parameters()
        maxSampleRates = {'single':{'8':1E9, '12': 500E6, '14':125E6}, 
                          'dual'  :{'8':1E9, '12': 500E6, '14':125E6},
                          'quad'  :{'8':1E9, '12': 500E6, '14':125E6}}
        maxRate = maxSampleRates[outputs][self.precision_bits]
        samplingPointsPerDiv = {'1K':50, '10K':500, '100K':5E3,'1M':50E3,
                                '10M':500E3,'25M':1.25E6,'50M':2.5E6,'100M':5E6,'250M':12.5E6}
        samplePts = samplingPointsPerDiv[self.sampling_points]
        if maxRate > samplePts / self.time_base:
            self.sample_rate = samplePts / self.time_base
        else:
            self.sample_rate =  maxRate
            
        return self.sample_rate
    
    def update_device_parameters(self,channel= 1):
        self.time_base = Quantity(self.query(
            self.configs['scope']['owon']['queries']['parameters']['general']['time_scale'])).real
        self.horizontal_offset = float(self.query(
            self.configs['scope']['owon']['queries']['parameters']['general']['horizontal_offset']))
        self.voltage_scale = Quantity(self.query(
            self.configs['scope']['owon']['queries']['parameters']['channel']['voltage_scale'].format(channel))).real
        self.vertical_offset = float(self.query(
            self.configs['scope']['owon']['queries']['parameters']['channel']['vertical_offset'].format(channel)))
        self.precision_bits = str(self.query(
            self.configs['scope']['owon']['queries']['parameters']['general']['memory_precision']))
        self.sampling_points = str(self.query(
            self.configs['scope']['owon']['queries']['parameters']['general']['memory_depth']))
    
    def capture(self):
        self.data_dict = {}
        for channel in self.channels:
            self.send(
                self.configs['scope']['owon']['commands']['capture']['begin'].format(channel))
            self.send(
                self.configs['scope']['owon']['commands']['capture']['wave_range'])
            self.data_dict[channel] = self.instrument.query_binary_values(
                self.configs['scope']['owon']['commands']['capture']['fetch'], datatype= 'h')
        self.send(
            self.configs['scope']['owon']['commands']['capture']['end'])
        _len = len(self.data_dict[1])
        if _len == 0:
            print("something went wrong")
        all_equal = all(len(self.data_dict[key]) == _len for key in self.data_dict)
        if not all_equal:
            print("something went wrong!")
            print("add debugging asap!")
        print("done! captured successfully")
            
    def calculate_voltage_and_time(self):
        self.scaled_data = {channel: [] for channel in self.channels}
        self.scaled_data['time_data'] = [i / self.sample_rate for i in range(len(self.data_dict[self.channels[0]]))]
        for channel in self.channels:
            self.scaled_data[channel] = [(float(item) / 6400 - self.channels_params[channel]['offset']) *
                                         self.voltage_scale for item in self.data_dict[channel]]

    def visualise(self, data):
        fig, ax = plt.subplots()
        ax.set(xlabel='time (S)', ylabel='voltage (V)', title='WAVEFORM')
        colours = ["-r", "-g", "-b", "-y"]
        for (channel,colour) in zip(self.channels, colours):
            ax.plot(data['time_data'], data[channel],
                    colour, label= "CH{}".format(channel))
        ax.grid()
        plt.legend(loc="upper left")
        plt.xlim([0, data['time_data'][-1]])
        plt.show()

    def extract_period_index_v1(self, wave_form):
        zero_buffer = 0.0001
        signal_buffer = 0.01
        index = 0
        for point in wave_form:
            if point > zero_buffer:
                index += 1
                continue
            first_zero_index = index
            break
        for point in wave_form[first_zero_index:]:
            if point <= zero_buffer:
                index += 1
                continue
            elif point >= signal_buffer:
                first_signal_index = index
                break
            else:
                print("unusual shape!!!!!!!!!")
            break
        for point in wave_form[first_signal_index:]:
            if point >= signal_buffer:
                index += 1
                continue
            elif point <= zero_buffer:
                second_zero_index = index
                break
            else:
                print("unusual shape!!!!!!!!!")
            break
        for point in wave_form[second_zero_index:]:
            if point <= zero_buffer:
                index += 1
                continue
            elif point >= signal_buffer:
                second_signal_index = index
                break
            else:
                print("unusual shape!!!!!!!!!")
            break
        for point in wave_form[second_signal_index:]:
            if point >= signal_buffer:
                index += 1
                continue
            elif point <= zero_buffer:
                third_zero_index = index
                break
            else:
                print("unusual shape!!!!!!!!!")
            break
        for point in wave_form[third_zero_index:]:
            if point <= zero_buffer:
                index += 1
                continue
            elif point >= signal_buffer:
                third_signal_index = index
                break
            else:
                print("unusual shape!!!!!!!!!")
            break      
        return [first_signal_index, third_signal_index]

    def extract_period_index_v2(self, wave_form):
        zero_buffer = 0.0001
        signal_buffer = self.wave_amplitude/2
        first_zero_index = next((i for i, point in enumerate(wave_form) if point <= zero_buffer), None)
        first_signal_index = next((i for i, point in enumerate(wave_form[first_zero_index:],
                                                               start=first_zero_index) if point >= signal_buffer), None)
        second_zero_index = next((i for i, point in enumerate(wave_form[first_signal_index:],
                                                              start=first_signal_index) if point <= zero_buffer), None)
        second_signal_index = next((i for i, point in enumerate(wave_form[second_zero_index:],
                                                                start=second_zero_index) if point >= signal_buffer), None)
        third_zero_index = next((i for i, point in enumerate(wave_form[second_signal_index:],
                                                              start=second_signal_index) if point <= zero_buffer), None)
        third_signal_index = next((i for i, point in enumerate(wave_form[third_zero_index:],
                                                                start=third_zero_index) if point >= signal_buffer), None)       
        if first_signal_index is None or second_signal_index is None or third_signal_index is None:
            print("unusual shape!!!!!!!!!")
            return []
        return [first_signal_index, third_signal_index]


    def extract_period_index_v3(self, wave_form, sample_rate, threshold= None):
        if threshold is None:
            threshold = self.wave_amplitude/2
        peaks, _ = find_peaks(wave_form, height= threshold, distance= sample_rate)
        if len(peaks) < 2:
            print("Not enough peaks found to determine a full period.")
            return []
        period_samples = peaks[1] - peaks[0]
        period_time = period_samples / sample_rate
        print("period time: ", period_time)
        one_period_waveform = wave_form[peaks[0]:peaks[0] + period_samples]
        return [peaks[0], peaks[0] + period_samples]

    def extract_period_index_v4(self, wave_form):
        zero_buffer = 0.0001
        signal_buffer = self.wave_amplitude/4
        first_zero_index = next((i for i, point in enumerate(wave_form) if point <= zero_buffer), None)
        first_signal_index = next((i for i, point in enumerate(wave_form[first_zero_index:],
                                                               start=first_zero_index) if point >= signal_buffer), None)
        second_signal_index = first_signal_index + int(self.sample_rate*self.wave_period)
        return [first_signal_index, second_signal_index]
        

    def clean_wave_form_data(self, initial_index, final_index):
            self.cleaned_data = {}
            for key in self.scaled_data:
                self.cleaned_data[key] = self.scaled_data[key][initial_index:final_index]
            time_shift = self.scaled_data['time_data'][initial_index]
            self.cleaned_data['time_data'] = list(np.array(self.cleaned_data['time_data']) - time_shift)
                
    def discriminator(self, cleaned_wave_form):
        type_one_wave = False
        mid_point = int(len(cleaned_wave_form)/2)
        zero_buffer = 0.001
        if np.average(cleaned_wave_form[mid_point-5:mid_point + 5]) > zero_buffer:
            type_one_wave = True
        return type_one_wave

    def qber_calculator(self):
        self.unix_time = time.time()
        cleaned_wave_form = self.cleaned_data[1]
        wave_size = len(cleaned_wave_form)
        wave_type_one = self.discriminator(cleaned_wave_form)
        first_index = 0
        second_index = int(wave_size * self.pduty1)
        if wave_type_one:
            third_index = int(wave_size * (self.pduty1 + self.nduty1))
            last_index = int(wave_size * (self.pduty1 + self.nduty1 + self.pduty2))
            H = np.average(self.cleaned_data[self.channels_dict['H']][first_index:second_index])
            V = np.average(self.cleaned_data[self.channels_dict['V']][first_index:second_index])
            self.hv_qber = V / (H + V)
            PLUS = np.average(self.cleaned_data[self.channels_dict['+']][third_index:last_index])
            MINUS = np.average(self.cleaned_data[self.channels_dict['-']][third_index:last_index])
            self.pm_qber = MINUS / (PLUS + MINUS)
            self.qber = self.hv_qber
        else:
            third_index = int(wave_size * (self.pduty1 + self.nduty2))
            last_index = int(wave_size * (self.pduty1 + self.nduty2 + self.pduty2))
            H = np.average(self.cleaned_data[self.channels_dict['H']][first_index:second_index])
            V = np.average(self.cleaned_data[self.channels_dict['V']][first_index:second_index])
            self.hv_qber = V / (H + V)
            PLUS = np.average(self.cleaned_data[self.channels_dict['+']][third_index:last_index])
            MINUS = np.average(self.cleaned_data[self.channels_dict['-']][third_index:last_index])
            self.pm_qber = MINUS / (PLUS + MINUS)
            self.qber = self.hv_qber

    def get_data(self):
        self.capture()
        self.calculate_voltage_and_time()
        test_wave_form = self.scaled_data[1]
        i_index, f_index = self.extract_period_index_v4(test_wave_form)
        self.clean_wave_form_data(i_index, f_index)
        self.qber_calculator()


    def update_data(self, additional_data):
        self.get_data()
        data_list = [self.unix_time, self.hv_qber, self.pm_qber, self.qber,
                     additional_data]
        for key, element in zip(self.result_dict.keys(), data_list):
            self.result_dict[key].append(element)

    def extract_results(self, output_name):
        df = pd.DataFrame(self.result_dict)
        df.to_csv(output_name, sep= ',')
