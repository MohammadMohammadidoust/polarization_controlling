import logging
import numpy as np
import time
import os

from keras.src.utils.module_utils import tensorflow

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#import tensorflow as tflow
from tensorflow.keras.layers import Dense, Activation, Input, Lambda
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam



logger = logging.getLogger(__name__)

@tensorflow.keras.utils.register_keras_serializable()
def combine_streams(inputs):
    v, a = inputs
    return v + (a - tensorflow.reduce_mean(a, axis=1, keepdims=True))

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims,)),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(n_actions)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


def build_dueling_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims, fcvalue_dims, fcadvantage_dims):
    input_layer = Input(shape=(input_dims,))
    dense1 = Dense(fc1_dims, activation='relu')(input_layer)
    dense2 = Dense(fc2_dims, activation='relu')(dense1)
    value_fc = Dense(fcvalue_dims, activation='relu')(dense2)
    value = Dense(1, activation=None)(value_fc)
    advantage_fc = Dense(fcadvantage_dims, activation='relu')(dense2)
    advantage = Dense(n_actions, activation=None)(advantage_fc)

    q_values = Lambda(combine_streams)([value, advantage])
    model = Model(inputs=input_layer, outputs=q_values)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

class Environment():
    def __init__(self, actions_space, acquire_polarization_instance,
                 polarization_controller_instance, qber_threshold):
        self.all_actions = {int(key): value for key, value in actions_space.items()}
        self.p_data_acquisition = acquire_polarization_instance
        self.p_controller = polarization_controller_instance
        #self.current_state= STATE
        self.terminal_condition= qber_threshold

    def translate_actions(self, action_index):
        action_string= self.all_actions[action_index]
        return list(action_string)

    def calculate_reward(self, boundary_condition):
        """
        Reward system is designed based on distance between current state and
        our expected state. I will punish agent in each step based on this
        distance and if agent finds expected state it gives a +10 points reward
        if agent tries to cross voltage threshold, it'll give a -10 point punish
        and if all voltages cross the thresholds, the episode will be finished.

        """

        if self.p_data_acquisition.qber < self.terminal_condition:
            logger.debug("Successful polarisation restoration")
            return 200.0, True
        if boundary_condition == 1:
            logger.debug("Hit the voltage boundaries")
            reward = -10 + (-2*(self.p_data_acquisition.qber - self.terminal_condition))
            return reward, False
        if boundary_condition == 2:
            logger.debug("Hit ALL voltage boundaries")
            reward = -30 + (-2*(self.p_data_acquisition.qber - self.terminal_condition))
            return reward, True
        return -2 * (self.p_data_acquisition.qber - self.terminal_condition), False

    def check_boundary_conditions(self, action):
        """
        function to check whether the action hits boundary conditions
        :return: three statuses: 0 means no boundaries are hit
        1 means at least one voltage and at most 3 voltage hit the boundary
        2 means all voltages hit the boundary
        """
        acts = self.translate_actions(action)
        if any(
            (volt >= self.p_controller.max_voltage and act != 'D') or
            (volt <= self.p_controller.min_voltage and act != 'U')
            for volt, act in zip(self.p_controller.current_voltages, acts)
            ):
            if all(
                (volt >= self.p_controller.max_voltage and act != 'D') or
                (volt <= self.p_controller.min_voltage and act != 'U')
                for volt, act in zip(self.p_controller.current_voltages, acts)
                ):
                return 2
            return 1
        return 0

    def step(self, action):
        boundary = self.check_boundary_conditions(action)
        if boundary == 2:
            reward, done= self.calculate_reward(boundary_condition= 2)
            self.p_controller.reset_voltages()
            self.p_data_acquisition.update_data([0, 0, 0, 0])
            state = np.append(self.p_controller.current_voltages.copy(),
                              [self.p_data_acquisition.qber])
            return (state, reward, done)
        else:
            acts = self.translate_actions(action)
            new_voltages = self.p_controller.action_to_voltages(acts)
            self.p_controller.send_voltages(new_voltages)
            time.sleep(0.4)         #response time
            self.p_data_acquisition.update_data(new_voltages)
            state = np.append(self.p_controller.current_voltages.copy(),
                              [self.p_data_acquisition.qber])
            reward, done= self.calculate_reward(boundary_condition= boundary)
            return state, reward, done

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete= False):
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
    def __init__(self, alpha, gamma, n_actions, discrete, epsilon,
                 batch_size, input_dims, epsilon_dec=0.996,
                 epsilon_end=0.01, fc1_dims= 256, fc2_dims= 256,
                 fcvalue_dims= 128, fcadvantage_dims= 128,
                 mem_size=1000000, fname='vanilla_dqn_model.h5',
                 fname2= 'double_dqn_target_model.h5',
                 model_type= "VanillaDQN", replace_target= 100):
        self.model_type = model_type
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.replace_target_cnt = replace_target
        self.learn_step_counter = 0
        self.model_file = fname
        self.target_model_file = fname2
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete)
        self.l_rate = alpha
        if model_type in ["VanillaDQN", "DoubleDQN"]:
            self.q_eval = build_dqn(alpha, n_actions, input_dims, fc1_dims, fc2_dims)
            self.q_target = build_dqn(alpha, n_actions, input_dims, fc1_dims, fc2_dims)
        elif model_type in ["DuelingDQN", "DoubleDuelingDQN"]:
            self.q_eval = build_dueling_dqn(alpha, n_actions, input_dims, fc1_dims, fc2_dims,
                                            fcvalue_dims, fcadvantage_dims)
            self.q_target = build_dueling_dqn(alpha, n_actions, input_dims, fc1_dims, fc2_dims,
                                              fcvalue_dims, fcadvantage_dims)
        else:
            logger.error("Model is Not Valid! check the model name again!")
            raise ValueError("Invalid model type selected.")
        self.update_target_network()

    def update_target_network(self):
        self.q_target.set_weights(self.q_eval.get_weights())    

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = np.atleast_2d(state)
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        q_eval = self.q_eval.predict(state)
        q_next = self.q_target.predict(new_state)
        if self.model_type in ["DoubleDQN", "DoubleDuelingDQN"]:
            q_eval_next = self.q_eval.predict(new_state)
            next_actions = np.argmax(q_eval_next, axis= 1)
            q_target_next = self.q_target.predict(new_state)
            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype= np.int32)
            q_target[batch_index, action_indices] = reward + \
                self.gamma * q_target_next[batch_index, next_actions] * done
        else:
            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype= np.int32)
            q_target[batch_index, action_indices] = reward + \
                self.gamma * np.max(q_next, axis= 1) * done
            
        _ = self.q_eval.fit(state, q_target, verbose=0)
        self.learn_step_counter += 1
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
            self.epsilon_min else self.epsilon_min
        if self.learn_step_counter% self.replace_target_cnt == 0:
            self.update_target_network()
        
    def save_model(self):
        self.q_eval.save(self.model_file)
        logger.info(f"Evaluation Model has been saved successfully in file: {self.model_file}")
        if self.model_type in ["DoubleDQN", "DoubleDuelingDQN"]:
            self.q_target.save(self.target_model_file)
            logger.info(f"Target Model has been saved successfully in file: {self.target_model_file}")

    def load_model(self, file_name):
        try:
            self.q_eval = load_model(file_name, compile= False)
            self.q_eval.compile(optimizer=Adam(learning_rate= self.l_rate), loss='mse')
            logger.info(f"Evaluation Model has been loaded successfully from file: {file_name}")
            print(f"Evaluation Model has been loaded successfully from file: {file_name}")
            if self.model_type in ["DoubleDQN", "DoubleDuelingDQN"]:
                try:
                    self.update_target_network()
                    print(f"Target Model has been loaded successfully updated")
                except:
                    logger.critical("Can not update Target Model! try again or start a new learning journey")
        except:
            logger.critical("Can not load Evaluation Model! try again or start a new learning journey")
            raise RuntimeError("Load model failed. Try again or Start without loading!")
