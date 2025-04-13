import pyvisa
import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantiphy import Quantity
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

class OWON:
    def __init__(self, conf_dict):
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

    def connect(self):
        """ the device will be connected automatically as pluged in, 
        this function is just for checking the connection and there is 
        no need to call it to connect to the device.
        """
        try:
            print("Oscilloscope is now connected!")
            print("device info: ", self.query('*IDN?'))
        except:
            logger.critical("OWON connection failed")
            print("can not connect to OWON Oscilloscope!")
            raise RuntimeError("Failed to connect OWON") 


    def auto_set_device(self):
        self.send(self.configs['scope']['owon']['commands']['autoset'])
        logger.debug("start autoset OWON device")
        print("Progress: ", self.query(
            self.configs['scope']['owon']['queries']['autoset_progress']))
        time.sleep(30)
        progress = int(self.query(
            self.configs['scope']['owon']['queries']['autoset_progress']))
        print("progress: ", progress)
        while progress < 100 :
            logger.warning("Auto setting OWON scope took longer than usual!")
            time.sleep(10)
        logger.debug("Auto set OWON is Done")

    def measure_params(self):
        logger.debug("Start measuring OWON parameters")
        self.send(
            self.configs['scope']['owon']['commands']['measure_params']['turn_on'])
        for channel in self.channels:
            source_command = self.configs['scope']['owon']['commands']['measure_params']['source']
            self.send(source_command.format(channel))
            self.channels_params[channel]['vmin'] = np.float64(
                self.query(self.configs['scope']['owon']['commands']['measure_params']['vmin']))
            self.channels_params[channel]['offset'] = self.channels_params[channel]['vmin'] / self.v_scale
        self.send(self.configs['scope']['owon']['commands']['measure_params']['turn_off'])

    def initialise(self):
        logger.debug("Start initialising OWON scope")
        for command in self.configs['scope']['owon']['commands']['initialise']['general']:
            self.send(command)
        for channel in self.channels:
            for command in self.configs['scope']['owon']['commands']['initialise']['channel']:
                self.send(command.format(channel))
        self.measure_params()

        time.sleep(10)
        logger.debug("OWON initialisation done!")

    def print_device_parameters(self):
        for general_query in self.configs['scope']['owon']['queries']['parameters']['general']:
                print(general_query, self.query(
                    self.configs['scope']['owon']['queries']['parameters']['general'][general_query]))
        for channel in self.channels:
            for channel_query in self.configs['scope']['owon']['queries']['parameters']['channel']:
                print(channel_query, self.query(
                    self.configs['scope']['owon']['queries']['parameters']['channel'][channel_query].format(channel)))

    def get_sample_rate(self, outputs):
        logger.debug("Start getting sample rate of OWON")
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
        logger.info(f"OWON current sample rate: {self.sample_rate}")
        return self.sample_rate
    
    def update_device_parameters(self,channel= 1):
        logger.debug("Start updating OWON parameters")
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
        logger.debug("Start Capturing data from OWON")
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
            logger.critical("OWON captured nothing!!!")
            raise RuntimeError("OWON couldn't has captured any data")
        all_equal = all(len(self.data_dict[key]) == _len for key in self.data_dict)
        if not all_equal:
            logger.critical("Inconsistent channel data during capture!!")
        logger.debug("done! captured successfully")
            
    def calculate_voltage_and_time(self):
        logger.debug("Start Scaling data from captured data")
        self.scaled_data = {channel: [] for channel in self.channels}
        self.scaled_data['time_data'] = [i / self.sample_rate for i in range(len(self.data_dict[self.channels[0]]))]
        for channel in self.channels:
            self.scaled_data[channel] = [(float(item) / 6400 - self.channels_params[channel]['offset']) *
                                         self.voltage_scale for item in self.data_dict[channel]]

    def visualise(self, data, test_form= False):
        logger.debug("Start visualising OWON data")
        fig, ax = plt.subplots()
        ax.set(xlabel='time (S)', ylabel='voltage (V)', title='WAVEFORM')
        if not test_form:
            colours = ["-r", "-g", "-b", "-y"]
            for (channel,colour) in zip(self.channels, colours):
                ax.plot(data['time_data'], data[channel], colour, label= "CH{}".format(channel))
            ax.grid()
            plt.legend(loc="upper left")
            plt.xlim([0, data['time_data'][-1]])
            plt.show()
        else:
            ax.plot(np.arange(len(data)), data, label= "visualising test wave form")
            plt.legend(loc="upper left")
            plt.show()


    def extract_period_index_v4(self, wave_form):
        logger.debug("Start extracting period from a waveform")
        zero_buffer = 0.001
        signal_buffer = self.wave_amplitude/5
        try: 
            first_zero_index = next((i for i, point in enumerate(wave_form) if point <= zero_buffer), None)
            first_signal_index = next((i for i, point in enumerate(wave_form[first_zero_index:],
                                                               start=first_zero_index) if point >= signal_buffer), None)
            second_signal_index = first_signal_index + int(self.sample_rate*self.wave_period)
            logger.debug("Period indices extracted successfully")
            return [first_signal_index, second_signal_index]
        except TypeError:
            logger.critical("Can not extract period indices!!!")
            self.visualise(wave_form, test_form= True)
            raise RuntimeError("Period incices can not be extracted from the waveform")
        
    def clean_wave_form_data(self, initial_index, final_index):
        logger.debug("Start cleaning waveform")
        self.cleaned_data = {}
        for key in self.scaled_data:
            self.cleaned_data[key] = self.scaled_data[key][initial_index:final_index]
        time_shift = self.scaled_data['time_data'][initial_index]
        self.cleaned_data['time_data'] = list(np.array(self.cleaned_data['time_data']) - time_shift)
                
    def discriminator(self, cleaned_wave_form):
        logger.debug("Start recognising wave type")
        type_one_wave = False
        mid_point = int(len(cleaned_wave_form)/2)
        zero_buffer = 0.001
        if np.average(cleaned_wave_form[mid_point-5:mid_point + 5]) > zero_buffer:
            type_one_wave = True
        logger.info(f"Type one Wave? {type_one_wave}")
        return type_one_wave

    def qber_calculator(self):
        logger.debug("Start calculating qber")
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
            PLUS = np.average(self.cleaned_data[self.channels_dict['+']][third_index:last_index])
            MINUS = np.average(self.cleaned_data[self.channels_dict['-']][third_index:last_index])
        else:
            third_index = int(wave_size * (self.pduty1 + self.nduty2))
            last_index = int(wave_size * (self.pduty1 + self.nduty2 + self.pduty2))
            H = np.average(self.cleaned_data[self.channels_dict['H']][third_index:last_index])
            V = np.average(self.cleaned_data[self.channels_dict['V']][third_index:last_index])
            PLUS = np.average(self.cleaned_data[self.channels_dict['+']][first_index:second_index])
            MINUS = np.average(self.cleaned_data[self.channels_dict['-']][first_index:second_index])
        self.hv_qber = V / (H + V)
        self.pm_qber = MINUS / (PLUS + MINUS)
        self.qber = (1 / np.sqrt(2)) * np.sqrt(self.hv_qber**2 + self.pm_qber**2)
        logger.debug(f"Current QBER is: {self.qber}")

    def get_data(self):
        logger.debug("Start acquiring data")
        self.capture()
        self.calculate_voltage_and_time()
        test_wave_form = self.scaled_data[2].copy()
        i_index, f_index = self.extract_period_index_v4(test_wave_form)
        self.clean_wave_form_data(i_index, f_index)
        self.qber_calculator()


    def update_data(self, additional_data):
        logger.debug("Start updating data")
        self.get_data()
        data_list = [self.unix_time, self.hv_qber, self.pm_qber, self.qber,
                     additional_data]
        for key, element in zip(self.result_dict.keys(), data_list):
            self.result_dict[key].append(element)

    def extract_results(self, output_name):
        logger.debug("Updated data has been written successfully!")
        df = pd.DataFrame(self.result_dict)
        df.to_csv(output_name, sep= ',')


class RIGOL:
    def __init__(self, conf_dict):
        self.configs = conf_dict
        self.hv_qber = None
        self.pm_qber = None
        self.qber = self.hv_qber
        self.resource = pyvisa.ResourceManager()
        self.resource_address = self.configs['scope']['rigol']['resource_address']
        self.instrument = self.resource.open_resource(self.resource_address)
        self.channels_dict = self.configs['scope']['rigol']['channels']
        self.channels = list(self.channels_dict.values())
        self.wave_amplitude = self.configs['scope']['rigol']['wave']['amplitude']
        self.wave_frequency = self.configs['scope']['rigol']['wave']['frequency']
        self.wave_period = 1.0 / self.wave_frequency
        self.pduty1 = self.configs['scope']['rigol']['wave']['pduty1']
        self.pduty2 = self.configs['scope']['rigol']['wave']['pduty2']
        self.nduty1 = self.configs['scope']['rigol']['wave']['nduty1']
        self.nduty2 = self.configs['scope']['rigol']['wave']['nduty2']
        self.result_dict = {"Unix_time": [], "hv_qber": [], "pm_qber": [],
                            "qber": [], "Additional_data": []}

    def send(self, command):
        self.instrument.write(str(command))

    def query(self, command, strip= True):
        if strip:
            response = self.instrument.query(str(command)).strip()
        else:
            response = self.instrument.query(str(command))
        return response

    def connect(self):
        """ the device will be connected automatically as pluged in, 
        this function is just for checking the connection and there is 
        no need to call it to connect to the device.
        """
        try:
            print("Oscilloscope is now connected!")
            print("device info", self.query('*IDN?'))
        except:
            logger.critical("RIGOL connection failed")
            print("can not connect to RIGOL Oscilloscope!")
            raise RuntimeError("Failed to connect OWON") 

    def auto_set_device(self):
        logger.debug("Start auto setting RIGOL")
        self.send(self.configs['scope']['rigol']['commands']['autoset'])
        time.sleep(5)
        logger.debug("RIGOL Auto set is done!")

    def initialise(self):
        logger.debug("Start Initialising RIGOL")
        for command in self.configs['scope']['rigol']['commands']['initialise']['general']:
            self.send(command)
        for channel in self.channels:
            for command in self.configs['scope']['rigol']['commands']['initialise']['channel']:
                self.send(command.format(channel))
        time.sleep(1)
        logger.debug("RIGOL initialise done!")

    def trigger_check(self):
        status = self.configs['scope']['rigol']['queries']['trigger_status']
        counter = 0
        while counter <= 500:
            state = self.query(status)
            if state == "STOP":
                return None
            counter += 1
            time.sleep(0.01)
        logger.critical("Something went wrong due to check trigger conditions")
        raise RuntimeError("Something went wrong due to check trigger conditions")

    def capture(self, attempt= 0, max_attempts= 5):
        self.scaled_data = {}
        self.send(self.configs['scope']['rigol']['commands']['stop_running'])
        self.trigger_check()
        for channel in self.channels:
            self.send(self.configs['scope']['rigol']['commands']['channel_source'].format(channel))
            preamble = self.query(self.configs['scope']['rigol']['queries']['preamble'],
                                  strip= False).split(',')
            #format_type   = int(preamble[0])
            #data_type     = int(preamble[1])
            #um_points    = int(preamble[2])
            #num_avg       = int(preamble[3])
            x_increment   = float(preamble[4])
            #x_origin      = float(preamble[5])
            x_reference   = float(preamble[6])
            y_increment   = float(preamble[7])
            y_origin      = float(preamble[8])
            y_reference   = float(preamble[9])

            self.send(self.configs['scope']['rigol']['queries']['acquire_data'])
            raw_data = self.instrument.read_raw()
            header_length = int(raw_data[1]) + 2
            raw_data = raw_data[header_length:-1]  # Remove header and terminator
            data = np.frombuffer(raw_data, dtype='B')
            voltage = (data - y_reference) * y_increment + y_origin
            self.scaled_data[channel] = voltage
            if channel == 1:
                time_axis = (np.arange(len(voltage)) - x_reference) * x_increment
                self.scaled_data['time_data'] = time_axis
        _len = len(self.scaled_data[1])
        all_equal = all(len(self.scaled_data[key]) == _len for key in self.scaled_data)
        if _len == 0 or not all_equal:
            logger.error("Something went wrong due to capturing data!")
            if attempt < max_attempts:
                logger.warning(f"Retrying capture (attempt {attempt + 1} of {max_attempts})...")
                self.send(self.configs['scope']['rigol']['commands']['start_running'])
                time.sleep(1)
                return self.capture(attempt=attempt + 1, max_attempts=max_attempts)
            else:
                print("Can not capture data!")
                logger.critical("Maximum capture attempts reached. Exiting capture.")
                raise RuntimeError("Maximum capture attempts reached. Exiting capture.")
        self.send(self.configs['scope']['rigol']['commands']['start_running'])          

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

    def extract_period_index(self):
        end = np.where(self.scaled_data['time_data'] >= self.wave_period)[0]
        start_index = 0
        stop_index = end[0]
        return [start_index, stop_index]

    def smoother(self):
        self.smoothed_data = {}
        for channel in self.channels:
            self.smoothed_data[channel] = savgol_filter(self.scaled_data[channel].copy(), 15, 1)
        self.smoothed_data['time_data'] = self.scaled_data['time_data'].copy()
            
    def extract_period_index_v4(self, s_channel):
        self.smoother()
        zero_buffer = 0.0001
        signal_buffer = 0.001
        first_zero_index = next((i for i, point in enumerate(self.smoothed_data[s_channel]) if point <= zero_buffer), None)
        try:
            first_signal_index = next((i for i, point in enumerate(self.smoothed_data[s_channel][first_zero_index:],
                                                                   start=first_zero_index) if point >= signal_buffer), None)
            end_time_value = self.smoothed_data['time_data'][first_signal_index] + self.wave_period
            second_signal_index = np.where(self.smoothed_data["time_data"] >= end_time_value)[0][0]
            return [first_signal_index, second_signal_index]
        except Exception as e:
            logger.error(e)
            self.visualise(self.smoothed_data)
            logger.critical("can not extract period indices!")
            raise RuntimeError("Period incices can not be extracted from the waveform")

    def clean_wave_form_data(self, initial_index, final_index):
        self.cleaned_data = {}
        for key in self.scaled_data:
            self.cleaned_data[key] = self.scaled_data[key][initial_index:final_index]
        time_shift = self.scaled_data['time_data'][initial_index]
        self.cleaned_data['time_data'] = np.array(self.cleaned_data['time_data']) - time_shift

    def discriminator(self, s_channel):
        type_one_wave = False
        mid_point = int(len(self.cleaned_data[s_channel])/2)
        zero_buffer = 0.001
        if np.average(self.cleaned_data[s_channel][mid_point-5:mid_point + 5]) > zero_buffer:
            type_one_wave = True
        return type_one_wave

    def qber_calculator(self, wave_type_one):
        self.unix_time = time.time()
        hv_qber_holder = self.hv_qber
        pm_qber_holder = self.pm_qber
        epsilon = 1e-10
        first_rise_time = self.wave_period * self.pduty1
        first_fall_time = self.wave_period * self.nduty1
        second_rise_time = self.wave_period * self.pduty2
        second_fall_time = self.wave_period * self.nduty2
        first_index = 0
        second_index = np.where(self.cleaned_data["time_data"] >= first_rise_time)[0][0]
        if wave_type_one:
            third_index = np.where(self.cleaned_data['time_data'] >=
                                   (first_rise_time + first_fall_time))[0][0]
            last_index = np.where(self.cleaned_data['time_data'] >=
                                  (first_rise_time + first_fall_time + second_rise_time))[0][0]
            H = np.average(self.cleaned_data[self.channels_dict['H']][first_index:second_index])
            V = np.average(self.cleaned_data[self.channels_dict['V']][first_index:second_index])
            PLUS = np.average(self.cleaned_data[self.channels_dict['+']][third_index:last_index])
            MINUS = np.average(self.cleaned_data[self.channels_dict['-']][third_index:last_index])
        else:
            third_index = np.where(self.cleaned_data['time_data'] >=
                                   (first_rise_time + second_fall_time))[0][0]
            last_index = np.where(self.cleaned_data['time_data'] >=
                                  (first_rise_time + second_fall_time + second_rise_time))[0][0]
            H = np.average(self.cleaned_data[self.channels_dict['H']][third_index:last_index])
            V = np.average(self.cleaned_data[self.channels_dict['V']][third_index:last_index])
            PLUS = np.average(self.cleaned_data[self.channels_dict['+']][first_index:second_index])
            MINUS = np.average(self.cleaned_data[self.channels_dict['-']][first_index:second_index])
        self.hv_qber = V / (H + V + epsilon)
        self.pm_qber = MINUS / (PLUS + MINUS + epsilon)
        if self.hv_qber >1 or self.hv_qber <0:
            self.hv_qber = hv_qber_holder
        if self.pm_qber >1 or self.pm_qber <0:
            self.pm_qber = pm_qber_holder
        self.qber = (1 / np.sqrt(2)) * np.sqrt(self.hv_qber**2 + self.pm_qber**2)
        logger.debug("Current QBER: {self.qber}")
        
    def get_data(self, source_channel= 1):
        self.capture()
        i_index, f_index = self.extract_period_index_v4(source_channel)
        self.clean_wave_form_data(i_index, f_index)
        type_one_wave = self.discriminator(source_channel)
        self.qber_calculator(type_one_wave)

    def update_data(self, additional_data, source_channel= 1):
        self.get_data(source_channel)
        data_list = [self.unix_time, self.hv_qber, self.pm_qber, self.qber,
                     additional_data]
        for key, element in zip(self.result_dict.keys(), data_list):
            self.result_dict[key].append(element)

    def extract_results(self, output_name):
        logger.debug("Updated data has been written successfully!")
        df = pd.DataFrame(self.result_dict)
        df.to_csv(output_name, sep= ',')
