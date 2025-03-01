import pyvisa
import serial
import numpy as np
import pandas as pd



class Thorlabs(object):
    def __init__(self, conf_dict):
        self.configs = conf_dict
        self.resource_address = self.configs['polarimeter']['thorlabs']['resource_address']
        self.mode = self.configs['polarimeter']['thorlabs']['mode']
        self.motor_on = self.configs['polarimeter']['thorlabs']['motor_on']
        self.motor_speed = self.configs['polarimeter']['thorlabs']['motor_speed']
        self.result_dict = {'Azimuth': [], 'Ellipticity': [], 'S1': [],
                          'S2': [], 'S3': [], 'QBER': [],
                          'Unix_time': [], 'Additional_data': []}
        
    def connect(self):
        rm = pyvisa.ResourceManager()
        self.device = rm.open_resource(self.resource_address)
        print("polarimeter connected: ", self.device.query('*IDN?'))
        self.device.write('{};:{};:{}'.format(self.mode, self.motor_on, self.motor_speed))
        print("Plarimeter Wavelength(m):", self.device.query(self.configs['polarimeter']['thorlabs']['queries']['wavelength']))
        print("Plarimeter Mode:", self.device.query(self.configs['polarimeter']['thorlabs']['queries']['mode']))
        print("Plarimeter Motor Speed(Hz):", self.device.query(self.configs['polarimeter']['thorlabs']['queries']['speed']))        
        
    def get_data(self):
        data = list(map(float, self.device.query(self.configs['polarimeter']['thorlabs']['queries']['acquire_data']).split(",")))
        timestamp = data[1]
        mode = data[2]
        self.az = data[9]*180/np.pi
        self.ellip = data[10]*180/np.pi
        dop = data[11]*100
        power = data[12]*1e3
        psi = data[9];
        chi = data[10];
        self.s1 = np.cos(2*Psi)*np.cos(2*Chi)
        self.s2 = np.sin(2*Psi)*np.cos(2*Chi)
        self.s3 = np.sin(2*Chi)
        self.unix_time = time.time()
        self.qber = (1 + self.s1)/2
                
    def update_data(self, additional_data):
        self.get_data()
        data_list = [self.az, self.ellip, self.s1,
                     self.s2, self.s3, self.qber,
                     self.unix_time, additional_data]
        for key, element in zip(self.result_dict.keys(), data_list):
            self.result_dict[key].append(element)
            
    def extract_results(self, output_name):
        df = pd.DataFrame(self.result_dict)
        df.to_csv(output_name, sep= ',')
