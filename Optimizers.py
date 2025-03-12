import numpy as np
import time

class PSO(object):
    """
    Particle Swarm Optimization (PSO) for minimizing QBER by adjusting optical fiber voltage.
    
    The code maintains:
      - A history of positions and velocities over iterations.
      - Personal best positions and scores for each particle.
      - A global best position and score.
    
    The evaluation loop is sequential (due to hardware limitations) and the velocity/position updates are vectorized.
    """
    
    def __init__(self, conf_dict, acquire_polarization_instance, polarization_controller_instance):
        self.configs = conf_dict
        self.learning_mode = self.configs['optimizer']['pso']['learning_mode']
        self.threshold = self.configs['optimizer']['pso']['qber_threshold']
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
    
    def run(self):
        self.reset_state()
        begin_time = time.perf_counter()
        self.position_x[0] = np.random.randint(self.min_x, self.max_x, (self.max_particles, self.dimensions))
        self.velocity[0] = np.random.randint(self.min_x, self.max_x, (self.max_particles, self.dimensions))
        self.personal_best_positions = self.position_x[0].copy()
        for iteration in range(self.max_iteration - 1):
            for particle_no in range(self.max_particles):
                current_position = self.position_x[iteration, particle_no, :]
                current_voltage = current_position.astype(int).tolist()
                self.p_controller.send_voltages(current_voltage)
                time.sleep(0.2)
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
                    print(f"Optimization finished at iteration {iteration} with total time: {total_time:.2f}s")
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
            print(f"Current QBER in iteration {iteration}: {self.p_data_acquisition.qber}")        
        print("Optimization finished without reaching the threshold.")
        total_time = time.perf_counter() - begin_time
        print(f"Total time: {total_time:.2f}s")


class SimulatedAnnealing():
    def __init__(self, conf_dict, acquire_polarization_instance, polarization_controller_instance):
        self.configs = conf_dict
        self.threshold = self.configs['optimizer']['sm']['qber_threshold']
        self.dimensions = self.configs['optimizer']['sm']['dimensions']
        self.bounds = self.configs['optimizer']['sm']['bounds']
        self.n_iterations = self.configs['optimizer']['sm']['n_iterations']
        self.step_size = self.configs['optimizer']['sm']['step_size']
        self.temp = self.configs['optimizer']['sm']['temp']
        self.low = self.configs['optimizer']['sm']['low']
        self.high = self.configs['optimizer']['sm']['high']
        self.p_controller = polarization_controller_instance
        self.p_data_acquisition = acquire_polarization_instance
        self.best = [0,0,0,0]

        def run():
            self.p_controller.send_voltages(self.best)
            self.p_data_acquisition.update_data(self.best)
            best_eval = self.p_data_acquisition.qber
            while best_eval > self.threshold:
                for dimension in range(self.dimensions):
                    best[dimension] = np.random.randint(low= self.low, high= self.high)
                self.p_controller.send_voltages(self.best)
                time.sleep(0.2)
                self.p_data_acquisition.update_data(self.best)
                best_eval = self.p_data_acquisition.qber
            curr, curr_eval = best, best_eval
            scores = []
            for i in range(self.n_iterations):
                candidate = curr + np.random.choice([-1, -0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7, 1],
                                                    size=len(self.bounds)) * self.step_size
                self.p_controller.send_voltages(candidate)
                time.sleep(0.2)
                self.p_data_acquisition.update_data(self.best)
                candidate_eval = self.p_data_acquisition.qber
                if candidate_eval < best_eval:
                    best, best_eval = candidate, candidate_eval
                    scores.append(best_eval)
                    print('>%d QBER(%s) = %.5f' % (i, best, best_eval))
                if best_eval < self.threshold:
                    break
                diff = candidate_eval - curr_eval
                t = temp / float(i + 1)
                metropolis = np.exp(-diff / t)
                if diff < 0 or np.random.rand() < metropolis:
                    curr, curr_eval = candidate, candidate_eval
            return [best, best_eval, scores]
