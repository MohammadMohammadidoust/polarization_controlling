import time
import logging
import numpy as np
from optimizers.DeepQNetwork import *

logger = logging.getLogger(__name__)

class PSO(object):
    """
    Particle Swarm Optimization (PSO) for minimizing QBER by adjusting optical fiber voltage.
    
    The code maintains:
      - A history of positions and velocities over iterations.
      - Personal best positions and scores for each particle.
      - A global best position and score.
    
    The evaluation loop is sequential (due to hardware limitations) 
    and the velocity/position updates are vectorized.
    """
    
    def __init__(self, conf_dict, acquire_polarization_instance, polarization_controller_instance):
        self.configs = conf_dict
        self.learning_mode = self.configs['optimizer']['pso']['learning_mode']
        self.threshold = self.configs['optimizer']['qber_threshold']
        self.min_x = self.configs['optimizer']['pso']['min_x']
        self.max_x = self.configs['optimizer']['pso']['max_x']
        self.max_particles = self.configs['optimizer']['pso']['max_particles']
        self.max_iteration = self.configs['optimizer']['pso']['max_iteration']
        self.weight = self.configs['optimizer']['pso']['weight']
        self.c1 = self.configs['optimizer']['pso']['c1']
        self.c2 = self.configs['optimizer']['pso']['c2']
        self.best_global_score = self.configs['optimizer']['pso']['qber_best_best']
        self.best_global_position = self.configs['optimizer']['pso']['voltage_best_best']
        self.initial_qber_best = self.configs['optimizer']['pso']['initial_qber_best']
        self.dimensions = self.configs['optimizer']['pso']['dimensions']
        self.p_controller = polarization_controller_instance
        self.p_data_acquisition = acquire_polarization_instance
        self.position_x = np.empty((self.max_iteration, self.max_particles, self.dimensions))
        self.velocity = np.empty((self.max_iteration, self.max_particles, self.dimensions))
        self.qber_values = np.empty((self.max_iteration, self.max_particles))
        self.personal_best_positions = np.empty((self.max_particles, self.dimensions))
        self.personal_best_scores = np.empty(self.max_particles)
        
    def reset_state(self):
        self.position_x.fill(0)
        self.velocity.fill(0)
        self.qber_values.fill(0)
        self.personal_best_positions.fill(0)
        self.personal_best_scores.fill(self.initial_qber_best)
        self.best_global_score = self.configs['optimizer']['pso']['qber_best_best']
        if self.learning_mode == "independent_learning":
            self.best_global_position = self.configs['optimizer']['pso']['voltage_best_best']
        logger.debug(f"Resetting PSO initial state done in mode {self.learning_mode}.")
    
    def run(self):
        logger.info("Start running PSO optimiser")
        self.reset_state()
        begin_time = time.perf_counter()
        self.position_x[0] = np.random.randint(self.min_x, self.max_x,
                                               (self.max_particles, self.dimensions))
        self.velocity[0] = np.random.randint(self.min_x, self.max_x,
                                             (self.max_particles, self.dimensions))
        self.personal_best_positions = self.position_x[0].copy()
        for iteration in range(self.max_iteration - 1):
            for particle_no in range(self.max_particles):
                current_position = self.position_x[iteration, particle_no, :]
                current_voltage = current_position.astype(int).tolist()
                self.p_controller.send_voltages(current_voltage)
                time.sleep(0.4)
                self.p_data_acquisition.update_data(current_voltage)
                current_qber = self.p_data_acquisition.qber
                self.qber_values[iteration, particle_no] = current_qber
                if current_qber <= self.personal_best_scores[particle_no]:
                    self.personal_best_scores[particle_no] = current_qber
                    self.personal_best_positions[particle_no] = current_position.copy()
                if current_qber <= self.best_global_score:
                    self.best_global_score = current_qber
                    self.best_global_position = current_position.copy()
                if self.best_global_score < self.threshold:
                    final_voltage = self.best_global_position.astype(int).tolist()
                    self.p_controller.send_voltages(final_voltage)
                    total_time = time.perf_counter() - begin_time
                    logger.info(f"Optimization finished at \
                    iteration {iteration} with total time: {total_time:.2f}s")
                    return
            r1 = np.random.rand(self.max_particles, self.dimensions)
            r2 = np.random.rand(self.max_particles, self.dimensions)
            new_velocity = (self.weight * self.velocity[iteration] +
                            self.c1 * r1 * (self.personal_best_positions - self.position_x[iteration]) +
                            self.c2 * r2 * (self.best_global_position - self.position_x[iteration]))
            new_position = self.position_x[iteration] + new_velocity
            new_position = np.clip(new_position, self.min_x, self.max_x)
            self.velocity[iteration + 1] = new_velocity
            self.position_x[iteration + 1] = new_position
            logger.debug(f"Current QBER in iteration {iteration}: {self.p_data_acquisition.qber}")        
        total_time = time.perf_counter() - begin_time
        logger.info(f"Optimization finished without reaching the threshold. \
        Total time: {total_time:.2f}s")


class SimulatedAnnealing():
    def __init__(self, conf_dict, acquire_polarization_instance, polarization_controller_instance):
        self.configs = conf_dict
        self.threshold = self.configs['optimizer']['qber_threshold']
        self.initial_threshold = self.configs['optimizer']['sa']['initial_threshold']
        self.dimensions = self.configs['optimizer']['sa']['dimensions']
        self.bounds = self.configs['optimizer']['sa']['bounds']
        self.n_iterations = self.configs['optimizer']['sa']['n_iterations']
        self.step_size = self.configs['optimizer']['sa']['step_size']
        self.temp = self.configs['optimizer']['sa']['temp']
        self.low = self.configs['optimizer']['sa']['low']
        self.high = self.configs['optimizer']['sa']['high']
        self.p_controller = polarization_controller_instance
        self.p_data_acquisition = acquire_polarization_instance
        self.best = [0,0,0,0]

    def run(self):
        logger.info("Start running SA optimiser")
        begin_time = time.perf_counter()
        self.p_controller.send_voltages(self.best)
        self.p_data_acquisition.update_data(self.best)
        best_eval = self.p_data_acquisition.qber
        while self.p_data_acquisition.qber > self.initial_threshold:
            for dimension in range(self.dimensions):
                self.best[dimension] = np.random.randint(low= self.low, high= self.high)
            self.p_controller.send_voltages(self.best)
            time.sleep(0.35)
            self.p_data_acquisition.update_data(self.best)
            best_eval = self.p_data_acquisition.qber
        curr, curr_eval = self.best, best_eval
        for i in range(self.n_iterations):
            candidate = curr + np.random.choice([-1, -0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7, 1],
                                                size=len(self.bounds)) * self.step_size
            self.p_controller.send_voltages(candidate)
            time.sleep(0.35)
            self.p_data_acquisition.update_data(candidate)
            candidate_eval = self.p_data_acquisition.qber
            if candidate_eval < best_eval:
                self.best, best_eval = candidate, candidate_eval
            if best_eval < self.threshold:
                self.p_controller.send_voltages(self.best)
                total_time = time.perf_counter() - begin_time
                logger.info(f"Optimization finished at \
                iteration {i} with total time: {total_time:.2f}s")
                break
            diff = candidate_eval - curr_eval
            t = self.temp / float(i + 1)
            metropolis = np.exp(-diff / t)
            if diff < 0 or np.random.rand() < metropolis:
                curr, curr_eval = candidate, candidate_eval
        total_time = time.perf_counter() - begin_time
        logger.info(f"Optimization finished without reaching the threshold. \
                    Total time: {total_time:.2f}s")




class DQN():
    def __init__(self, conf_dict, acquire_polarization_instance, polarization_controller_instance):
        self.configs = conf_dict
        self.model_type = self.configs['optimizer']['dqn']['model_type']
        self.total_runs = self.configs['optimizer']['dqn']['total_runs']
        self.episode = 0
        self.scores = []
        self.all_actions = self.configs['optimizer']['dqn']['all_actions']
        self.qber_threshold = self.configs['optimizer']['qber_threshold']
        self.mem_size = self.configs['optimizer']['dqn']['memory_size']
        self.discrete = self.configs['optimizer']['dqn']['discrete_actions_space']
        self.input_dims = self.configs['optimizer']['dqn']['input_dims']
        self.n_actions = self.configs['optimizer']['dqn']['n_actions']
        self.learning_rate = self.configs['optimizer']['dqn']['learning_rate']
        self.fc1_dims = self.configs['optimizer']['dqn']['fc1_dims']
        self.fc2_dims = self.configs['optimizer']['dqn']['fc2_dims']
        self.fc3_dims = self.configs['optimizer']['dqn']['fcvalue_dims']
        self.fc4_dims = self.configs['optimizer']['dqn']['fcadvantage_dims']
        self.gamma = self.configs['optimizer']['dqn']['gamma']
        self.epsilon = self.configs['optimizer']['dqn']['epsilon']
        self.epsilon_dec = self.configs['optimizer']['dqn']['epsilon_dec']
        self.epsilon_end = self.configs['optimizer']['dqn']['epsilon_end']
        self.batch_size = self.configs['optimizer']['dqn']['batch_size']
        self.model_file = self.configs['optimizer']['dqn']['fname']
        self.model_file2 = self.configs['optimizer']['dqn']['fname2']
        self.replace_target = self.configs['optimizer']['dqn']['replace_target']
        self.env = Environment(self.all_actions, acquire_polarization_instance,
                               polarization_controller_instance, self.qber_threshold)
        self.agent = Agent(self.learning_rate, self.gamma, self.n_actions, self.discrete,
                           self.epsilon, self.batch_size, self.input_dims, self.epsilon_dec,
                           self.epsilon_end, self.fc1_dims, self.fc2_dims, self.fc3_dims,
                           self.fc4_dims, self.mem_size, self.model_file, self.model_file2,
                           self.model_type, self.replace_target)

    def run(self):
        self.env.p_data_acquisition.update_data(self.env.p_controller.current_voltages)
        observation = self.env.p_data_acquisition.qber
        done = False
        score = 0
        while not done:
            action = self.agent.choose_action(observation)
            print("action: ", action)
            observation_, reward, done= self.env.step(action)
            score += reward
            print("score: ", score)
            self.agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            self.agent.learn()
        self.scores.append(score)
        self.episode += 1
        if self.episode % 50 == 0:
            avg_score = np.mean(self.scores[max(0, self.episode - 100):(self.episode + 1)])
            logging.info(f"Episode: {self.episode} Average Scores: {avg_score}")
            self.agent.save_model()
            
    def load_model(self):
        self.agent.load_model()
    
