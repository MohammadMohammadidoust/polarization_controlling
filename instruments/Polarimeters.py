import time
import pyvisa
import serial
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
        try: 
            rm = pyvisa.ResourceManager()
            self.device = rm.open_resource(self.resource_address)
            self.device.write('{};:{};:{}'.format(self.mode, self.motor_on, self.motor_speed))
            logger.info("polarimeter connected: {}".format(self.device.query('*IDN?')))
            print("Polarimeter is know connected!")
            print("Device info: {}".format(self.device.query('*IDN?')))
        except:
            logger.critical("ThorLabs connection failed")
            print("can not connect to ThorLabs Polarimeter!")
            raise RuntimeError("Failed to connect ThorLabs") 

        
    def get_data(self):
        logger.debug("Start getting data from ThorLabs")
        data = list(map(float, self.device.query(self.configs['polarimeter']['thorlabs']['queries']['acquire_data']).split(",")))
        timestamp = data[1]
        mode = data[2]
        self.az = data[9]*180/np.pi
        self.ellip = data[10]*180/np.pi
        dop = data[11]*100
        power = data[12]*1e3
        psi = data[9];
        chi = data[10];
        self.s1 = np.cos(2*psi)*np.cos(2*chi)
        self.s2 = np.sin(2*psi)*np.cos(2*chi)
        self.s3 = np.sin(2*chi)
        self.unix_time = time.time()
        self.qber = (1 + self.s1)/2
        logger.info("Current QBER: {}".fromat(self.qber))
                
    def update_data(self, additional_data):
        self.get_data()
        data_list = [self.az, self.ellip, self.s1,
                     self.s2, self.s3, self.qber,
                     self.unix_time, additional_data]
        for key, element in zip(self.result_dict.keys(), data_list):
            self.result_dict[key].append(element)
            
    def extract_results(self, output_name):
        logger.debug("Updated data has been written successfully!")
        df = pd.DataFrame(self.result_dict)
        df.to_csv(output_name, sep= ',')
