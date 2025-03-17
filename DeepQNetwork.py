import logging
import pyvisa
import serial
import pandas as pd
import numpy as np
import time
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import hashlib
import datetime
import sys


class Environment():
    def __init__(self, conf_dict, acquire_polarization_instance, polarization_controller_instance):
        self.configs = conf_dict
        self.all_actions= {0: "ZZZZ", 1: "ZZZU", 2: "ZZZD", 3: "ZZUZ", 4: "ZZDZ",
                           5: "ZUZZ", 6: "ZDZZ", 7: "UZZZ", 8: "DZZZ", 9: "ZZUU",
                           10: "ZZDD", 11: "ZZUD", 12: "ZZDU", 13: "ZUZU",
                           14: "ZDZD", 15: "ZUZD", 16: "ZDZU", 17: "UZZU",
                           18: "DZZD", 19: "UZZD", 20: "DZZU", 21: "ZUUZ",
                           22: "ZDDZ", 23: "ZUDZ", 24: "ZDUZ", 25: "UZUZ",
                           26: "DZDZ", 27: "UZDZ", 28: "DZUZ", 29: "UUZZ",
                           30: "DDZZ", 31: "UDZZ", 32: "DUZZ", 33: "ZUUU",
                           34: "ZDDD", 35: "ZUUD", 36: "ZUDU", 37: "ZDUU",
                           38: "ZDDU", 39: "ZDUD", 40: "ZUDD", 41: "UUUZ",
                           42: "DDDZ", 43: "UUDZ", 44: "UDUZ", 45: "DUUZ",
                           46: "DDUZ", 47: "DUDZ", 48: "UDDZ", 49: "UZUU",
                           50: "DZDD", 51: "UZUD", 52: "UZDU", 53: "DZUU",
                           54: "DZDU", 55: "DZUD", 56: "UZDD", 57: "UUZU",
                           58: "DDZD", 59: "UUZD", 60: "UDZU", 61: "DUZU",
                           62: "DDZU", 63: "DUZD", 64: "UDZD", 65: "UUUU",
                           66: "DDDD", 67: "UUUD", 68: "UUDU", 69: "UDUU",
                           70: "DUUU", 71: "UUDD", 72: "DDUU", 73: "UDUD",
                           74: "DUDU", 75: "DUUD", 76: "UDDU", 77: "DDDU",
                           78: "DDUD", 79: "DUDD", 80: "UDDD" }
        self.action_indices= list(self.all_actions.keys())
        self.current_state= STATE
        self.terminal_condition= QBER_threshold          # finishing state
        self.data_dict = {"S1": [], "S2": [], "S3": [], "AZ": [], "ELIP": [],
                          "unix_time": []}
        self.output_name = outputfile
        for _ in range(self.shocker_number):
            self.shockers.append(Shocker())

    def update_state(self):
        inputs = Acquire_Data()
        self.current_state = inputs[0:3]
        for key, element in zip(self.data_dict.keys(), inputs):
            self.data_dict[key].append(element)

    def send_voltages(self):
        voltages= [shocker.voltage for shocker in self.shockers]
        if Polarization_Controller.isOpen():
            Polarization_Controller.write(("V1,"+str(int(voltages[0]))+"\r\n").encode('ascii'))
            Polarization_Controller.write(("V2,"+str(int(voltages[1]))+"\r\n").encode('ascii'))
            Polarization_Controller.write(("V3,"+str(int(voltages[2]))+"\r\n").encode('ascii'))
            Polarization_Controller.write(("V4,"+str(int(voltages[3]))+"\r\n").encode('ascii'))
        else:
            print("Polarization Controler is not open")

    def turn_off_shockers(self):
        for shocker in self.shockers:
            shocker.voltage= 0

    def extract_data(self):
        df = pd.DataFrame(self.data_dict)
        df.to_csv(self.output_name, sep= ',')

    def translate_actions(self, action_index):
        action_string= self.all_actions[action_index]
        return list(action_string)

    def calculate_reward(self, state, boundry_condition):
        ### Reward system is designed based on distance between current state and
        ### our expected state. I will punish agent in each step based on this
        ### distance and if agent finds expected state it gives a +10 points reward
        ### if agent tries to cross voltage treshold, it'll give a -2000 punish
        qber= QBER(state)
        if boundry_condition is True:
            return (-2000, True)                             #reward and done status
        if qber < self.terminal_condition:
            return (10.0, True)                             #reward and done status
        return (-2*(qber - self.terminal_condition), False)  #reward and done status

    def check_boundry_conditions(self, action):
        acts= self.translate_actions(action)
        for i in range(len(acts)):
            if ((self.shockers[i].voltage >= self.shockers[i].max_voltage and
                 acts[i] != 'D') or
                (self.shockers[i].voltage <= self.shockers[i].min_voltage and
                 acts[i] != 'U')):
                return True
            return False

    def step(self, action):
        boundry= self.check_boundry_conditions(action)
        if boundry:
            reward, done= self.calculate_reward(self.current_state, boundry_condition= True)
            for shock in (self.shockers):
                shock.reset_voltage()
            self.send_voltages()
            return (self.current_state, reward, done)
        else:
            acts= self.translate_actions(action)
            for i, shock in enumerate(self.shockers):
                shock.update_voltage(acts[i])
            self.send_voltages()
            time.sleep(0.3)         #response time
            self.update_state()
            reward, done= self.calculate_reward(self.current_state, boundry_condition= False)
            return (np.array([self.current_state]), reward, done)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        if isinstance(state, list):
            state = np.array(state)
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)
            q_eval = self.q_eval.predict(state)
            q_next = self.q_eval.predict(new_state)
            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = reward + \
                                  self.gamma*np.max(q_next, axis=1)*done
            _ = self.q_eval.fit(state, q_target, verbose=0)
            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min
    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
