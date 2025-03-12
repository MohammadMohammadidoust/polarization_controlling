import numpy as np
import time

class PSO(object):
    def __init__(self, conf_dict, acquire_polarization_instance,
                 polarization_controller_instance):
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
        self.qber_best_best = self.configs['optimizer']['pso']['qber_best_best']
        self.voltage_best_best = self.configs['optimizer']['pso']['voltage_best_best']
        self.initial_qber_best = self.configs['optimizer']['pso']['initial_qber_best']
        self.dimensions = self.configs['optimizer']['pso']['dimensions']
        self.p_controller = polarization_controller_instance
        self.p_data_acquisition = acquire_polarization_instance
        self.position_x = np.empty([self.max_iteration, self.max_particles, self.dimensions])
        self.velocity = np.empty([self.max_iteration, self.max_particles, self.dimensions])
        self.qber_values = np.empty([self.max_iteration, self.max_particles])
        self.qber_best = np.empty([self.max_particles])
        self.voltage_best = np.empty([self.max_iteration, self.max_particles, self.dimensions])
        

    def reset_state(self):
        self.qber_values = np.empty([self.max_iteration, self.max_particles])
        self.position_x = np.empty([self.max_iteration, self.max_particles, self.dimensions])
        self.velocity = np.empty([self.max_iteration, self.max_particles, self.dimensions])
        self.qber_best = np.empty([self.max_particles])
        self.voltage_best = np.empty([self.max_iteration, self.max_particles, self.dimensions])
        self.qber_best_best = self.configs['optimizer']['pso']['qber_best_best']
        if self.learning_mode == "independent_learning":
            self.voltage_best_best = self.configs['optimizer']['pso']['voltage_best_best']
            
        

    def run(self):
        self.reset_state()
        begin_time = time.perf_counter()
        iteration = 0
        flag = 0
        for particle_no in range(self.max_particles):
            for dimension in range(self.dimensions):
                self.position_x[iteration][particle_no][dimension] = np.random.randint(low= self.min_x, high= self.max_x)
                self.velocity[iteration][particle_no][dimension]= np.random.randint(low= self.min_x, high= self.max_x)
            self.qber_best[particle_no] = self.initial_qber_best # 50% QBER
        while iteration < (self.max_iteration - 1):
            for paticle_no in range(self.max_particles):
                voltages = [self.position_x[iteration][particle_no][i] for i in range(self.dimensions)]
                self.p_controller.send_voltages(voltages)
                time.sleep(0.2)
                self.p_data_acquisition.update_data(voltages)
                self.qber_values[iteration][particle_no] = self.p_data_acquisition.qber
                if self.qber_values[iteration][particle_no] <= self.qber_best[particle_no]:
                    self.qber_best[particle_no] = self.qber_values[iteration][particle_no]
                    self.voltage_best[iteration][particle_no] = self.position_x[iteration][particle_no]
                if self.qber_values[iteration][particle_no] <= self.qber_best_best:
                    self.qber_best_best = self.qber_values[iteration][particle_no]
                    self.voltage_best_best = self.position_x[iteration][particle_no]
                if self.qber_best_best < self.threshold:
                    flag = 1
                    self.p_controller.send_voltages([self.voltage_best_best[i]
                                                     for i in range(self.dimensions)])
                    total_time = time.perf_counter() - begin_time
                    #print("Iteration number=", iteration)
                    #print("Total Time(s)=", total_time)
                    #print("Voltage Point(mV)=", self.voltage_best_best)
                    #print("Minimum QBER=", self.qber_best_best)
                    break
                for dimension in range(self.dimensions):
                    r1, r2 = np.random.choice([0, 1], size= 2)
                    self.velocity[iteration + 1][particle_no][dimension] = self.weight * self.velocity[iteration][particle_no][dimension] + self.c1 * r1 * (self.voltage_best[iteration][particle_no][dimension] - self.position_x[iteration][particle_no][dimension]) + self.c2 * r2 * (self.voltage_best_best[dimension] - self.position_x[iteration][particle_no][dimension])
                    self.position_x[iteration + 1][particle_no][dimension] = self.position_x[iteration][particle_no][dimension] + self.velocity[iteration + 1][particle_no][dimension]
                    if self.position_x[iteration + 1][particle_no][dimension] > self.max_x:
                        self.position_x[iteration + 1][particle_no][dimension] = self.max_x
                    if self.position_x[iteration + 1][particle_no][dimension] < self.min_x:
                        self.position_x[iteration + 1][particle_no][dimension] = self.min_x
                try:
                    self.position_x[iteration + 1][particle_no]=[int(self.position_x[iteration + 1][particle_no][i])
                                                                 for i in range(self.dimensions)]
                except:
                    self.position_x[iteration + 1][particle_no] = [0, 0, 0, 0]
                    print("error")
                    break
                
            if flag == 1:
                break
            print("Current QBER inside PSO running: ", self.p_data_acquisition.qber)
            iteration += 1
        print("Optimisation has been finished!")


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
