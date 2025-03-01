"""
   PSO_automatic_data_saving.py

   
   Quantum Communication Group, Iranian Center for Quantum Technologies (ICQTs)
   modified: 05 March 2023

   Explain: Python code to run a particle swarm optimization (PSO) algorithm for minimizing
   the quantum bit error rate (QBER) of a quantum key distribution (QKD) system.

"""
import logging
import pyvisa
import serial
import sys
import json
import time
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation


logging_path = 'hamid.log'
logging_format = '%(asctime)s : %(levelname)s : %(message)s'
logging_level = logging.INFO
logging.basicConfig(filename= logging_path, format= logging_format, level= logging_level)

with open('config.json', 'r') as jfile:
    CONFIG = json.load(jfile)

START_TIME = time.perf_counter()
RUNNING_TIME = CONFIG['running_time']
QBER_THRESHOLD = CONFIG['QBER_threshold']

    
class AcquireData(object):
    def __init__(self, calculation_mode= 'polarimeter', output_file= 'results.csv'):
        self.calc_mode = calculation_mode
        self.polarimeter_port = CONFIG['polarimeter_data']['port']
        self.polarimeter_mode = CONFIG['polarimeter_data']['mode']
        self.polarimeter_motor_on = CONFIG['polarimeter_data']['motor_on']
        self.polarimeter_motor_speed = CONFIG['polarimeter_data']['motor_speed']
        self.output_name = output_file
        if self.calc_mode = 'polarimeter':
            self.data_dict = {'Azimuth': [], 'Ellipticity': [], 'S1': [],
                              'S2': [], 'S3': [], 'QBER': [],
                              'Unix_time': [], 'Additional_data': []}

    def connect_polarimeter(self):
        rm = pyvisa.ResourceManager()
        self.polarimeter = rm.open_resource(rm.list_resources()[1])
        print(self.polarimeter.query('*IDN?'))
        self.polarimeter.write('{};:{};:{}'.format(self.polarimeter_port, self.polarimeter_motor_on,
                                                   self.polarimeter_motor_speed))
        print("Plarimeter Wavelength(m):", self.polarimeter.query('SENS:CORR:WAV?'))
        print("Plarimeter Mode:", self.polarimeter.query('SENS:CALC?'))
        print("Plarimeter Motor Speed(Hz):", self.polarimeter.query(':INP:ROT:VEL?'))

    def connect_polarization_controller(self):
        self.polarization_controller = serial.Serial(CONFIG['p_controller']['port'],
                                                     CONFIG['p_controller']['ff'],
                                                     timeout= CONFIG['p_controller']['tout'])

    def send_voltages_to_polarization_controller(self, volts):
        if self.polarization_controller.isOpen():
            pc.write(("V1,"+str(int(volts[0]))+"\r\n").encode('ascii'))
            pc.write(("V2,"+str(int(volts[1]))+"\r\n").encode('ascii'))
            pc.write(("V3,"+str(int(volts[2]))+"\r\n").encode('ascii'))
            pc.write(("V4,"+str(int(volts[3]))+"\r\n").encode('ascii'))
        else:
            print("Polarization Controller is not open!")

        
        
            
    def get_polarimeter_data(self):
        polarimeter_data = list(map(float, self.polarimeter.query('SENS:DATA:LAT?').split(",")))
        timestamp = polarimeter_data[1]
        mode = polarimeter_data[2]
        self.az = polarimeter_data[9]*180/np.pi
        self.ellip = polarimeter_data[10]*180/np.pi
        dop = polarimeter_data[11]*100
        power = polarimeter_data[12]*1e3
        psi = polarimeter_data[9];
        chi = polarimeter_data[10];
        self.s1 = np.cos(2*Psi)*np.cos(2*Chi)
        self.s2 = np.sin(2*Psi)*np.cos(2*Chi)
        self.s3 = np.sin(2*Chi)
        self.unix_time = time.time()


    def get_time_controller_data(self):
        pass

    def calculate_qber(self):
        if self.calc_mode = 'polarimeter':
            self.qber = (1 + self.s1)/2
            return self.qber
        else:
            pass

    def update_written_data(self, additional_data):
        if calc_mode = 'polarimeter':
            data_list = [self.az, self.ellip, self.s1,
                         self.s2, self.s3, self.qber,
                         self.unix_time, additional_data]
            for key, element in zip(self.data_dict.keys(), data_list):
                self.data_dict[key].append(element)
                
    def extract_results(self):
        df = pd.DataFrame(self.data_dict)
        df.to_csv(self.output_name, sep= ',')
            


class Optimizer(object):
    def __init__(self, q_thres, optimizer= 'PSO', calculation_mode= 'polarimeter'):
        self.optimizer = optimizer
        self.calc_mode = calculation_mode
        self.initialiser = AcquireData(calculation_mode= CONFIG['calc_mode'],
                                       output_file= CONFIG['output_file'])
        self.qber_threshold = QBER_THRESHOLD

    def initialise(self):
        if self.calc_mode = 'polarimeter':
            self.initialiser.connect_polarization_controller()
            self.initialiser.connect_polarimeter()
            self.get_polarimeter_data()
            self.initialiser.update_written_data([-1, -1, -1, -1])
        else:
            pass
        

    def pso_optimizer(self, max_particles, max_iteration, weight, c1, c2,
                      qber_best_best, voltage_best_best, initial_qber_best,
                      dimensions, max_x, min_x):
        position_x = np.empty([max_iteration, max_particles, dimensions])
        velocity = np.empty([max_iteration, max_particles, dimensions])
        qber_values = np.empty([max_iteration, max_particles])
        qber_best = np.empty([max_particles])
        voltage_best = np.empty([max_iteration, max_particles, dimensions])
        begin_time = time.perf_counter()
        iteration = 0
        flag = 0

        for particle_no in range(max_particles):
            for dimension in range(dimensions):
                position_x[iteration][particle_no][dimension] = np.random.randint(low= min_x, high= max_x)
                velocity[iteration][particle_no][dimension]= np.random.randint(low= min_x, high= max_x)
            qber_best[particle_no] = initial_qber_best # 50% QBER
        while iteration < (max_iteration - 1):
            for paticle_no in range(max_particles):
                self.initialiser.send_voltages_to_polarization_controller([position_x[iteration][particle_no][i]
                                                                           for i in range(dimensions)])
                time.sleep(0.5)
                self.initialiser.get_polarimeter_data()
                qber_values[iteration][particle_no] = self.initialiser.calculate_qber()
                voltages = [v1, v2, v3, v4]
                self.initialiser.update_written_data(voltages)
                if qber_values[iteration][particle_no] <= qber_best[particle_no]:
                    qber_best[particle_no] = qber_values[iteration][particle_no]
                    voltage_best[iteration][particle_no] = position_x[iteration][particle_no]
                if qber_values[iteration][particle_no] <= qber_best_best:
                    qber_best_best = qber_values[iteration][particle_no]
                    voltage_best_best = position_x[iteration][particle_no]
                if qber_best_best < self.qber_threshold:
                    flag = 1
                    self.initialiser.send_voltages_to_polarization_controller([voltage_best_best[i]
                                                                               for i in range(dimensions)])
                    total_time = time.perf_counter() - begin_time
                    print("Iteration number=", iteration)
                    print("Total Time(s)=", total_time)
                    print("Voltage Point(mV)=", voltage_best_best)
                    print("Minimum QBER=", qber_best_best)
                    break
                for dimension in range(dimensions):
                    r1, r2 = np.random.choice([0, 1], size= 2)
                    velocity[iteration + 1][particle_no][dimension] = weight *
                    velocity[iteration][particle_no][dimension] + c1 * r1 *
                    (voltage_best[iteration][particle_no][dimension] - position_x[iteration][particle_no][dimension]) +
                    c2 * r2 * (voltage_best_best[dimension] - position_x[iteration][particle_no][dimension])
                    position_x[iteration + 1][particle_no][dimension] = position_x[iteration][particle_no][dimension] +
                    velocity[iteration + 1][particle_no][dimension]
                    if position_x[iteration + 1][particlo_no][dimension] > max_x:
                        position_x[iteration + 1][particlo_no][dimension] = max_x
                    if position_x[iteration + 1][particlo_no][dimension] < min_x:
                        position_x[iteration + 1][particlo_no][dimension] = min_x
                try:
                    position_x[iteration + 1][particle_no]=[int(position_x[iteration + 1][particle_no][i]) for i in range(dimensions)]
                except:
                    position_x[iteration + 1][particle_no] = [0, 0, 0, 0]
                    print("error")
                    break
                
            if flag == 1:
                break
            print("Current QBER: ", qber_best_best)
            iteration += 1





    def animate(self, xs, ys):
        if (self.calc_mode == 'polarization' && self.optimizer == 'PSO'):
            if time.perf_counter() - START_TIME > RUNNING_TIME:
                print("Finished!")
                self.initialiser.extract_results()
                sys.exit()
            self.initialiser.get_polarimeter_data()
            self.initialiser.update_written_data([-1, -1, -1, -1])
            qber = self.initialiser.calculate_qber()
            if qber > self.qber_threshold:
                self.pso_optimizer(max_particles= CONFIG['PSO_config']['max_particles'],
                                   max_iteration= CONFIG['PSO_config']['max_iteration'],
                                   weight= CONFIG['PSO_config']['weight'], dimensions= CONFIG['PSO_config']['dimensions'],
                                   c1= CONFIG['PSO_config']['c1'], c2= CONFIG['PSO_config']['c2'],
                                   initial_qber_best= CONFIG['PSO_config']['initial_qber_best'],
                                   qber_best_best= CONFIG['PSO_config']['qber_best_best'],
                                   voltage_best_best= CONFIG['PSO_config']['voltage_best_best'],
                                   max_x= CONFIG['PSO_config']['max_x'], min_x= CONFIG['PSO_config']['min_x'])
            Y = self.initialiser.qber * 100
            xs.append(dt.datetime.now().strftime('%S.%f'))
            ys.append(Y)
            xs = xs[-20:]
            ys = ys[-20:]
            ax.clear()
            ax.plot(xs, ys)
            ax.set_ylim([0, 100])
            plt.xticks(rotation= 45, ha= 'right')
            plt.subplots_adjust(bottom= 0.30)
            plt.title('QBER Value over Time')
            plt.ylabel('QBER (%)')


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

op = Optimizer(q_thres = QBER_THRESHOLD, optimizer= CONFIG['optimizer'],
               calculation_mode= CONFIG['calc_mode'])
ani = animation.FuncAnimation(fig, op.animate, fargs=(xs, ys), interval=100)
plt.show()
