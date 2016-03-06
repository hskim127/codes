import numpy as np
import random
from numpy import linalg as LA

class FirstOrderMethod:
    def __init__(self,problem,method,parameters):
        # Problem instance
        self.problem = problem

        # Optimization method
        self.method = method

        # Information from the iteration.
        self.results_stored = parameters['result_stored']
        self.results = {'trn_loss':[],'tst_loss':[],'trn_err':[],'tst_err':[],'norm_grad':[],'n_grad_comp':[]}

        # Step size
        self.eta = parameters["step-size"]
        # Maximum passes over data
        self.max_pass = parameters["# max-pass"]
        # Epoch-size
        if "epoch-size" in parameters.keys():
            self.epoch_size = parameters["epoch-size"]
        
        # Error (Norm of gradient)
        self.epsilon = parameters["epsilon"]

    def do_optimization(self):
        if self.method == 'fgd':
            self.full_gradient_descent()
        elif self.method == 'sgd':
            self.stochastic_gradient_descent()
        elif self.method == 'svrg':
            self.svrg()

    def full_gradient_descent(self):
        n_samples = self.problem.get_n_sample()
        max_T = n_samples*self.max_pass
        t = 0
        while True:
            # Compute the component gradient
            g = self.problem.compute_full_gradient()

            # Save the training results
            self.compute_iteration_info(g,t)

            # Check the stopping criterion
            if t > max_T or self.results['norm_grad'][-1] < self.epsilon:
                break

            # Gradient update
            self.problem.update(-self.eta*g)
            t += n_samples

            
    def stochastic_gradient_descent(self):
        n_samples = self.problem.get_n_sample()
        max_T = n_samples*self.max_pass
        t = 0

        # Save the training results
        self.compute_iteration_info(self.problem.compute_full_gradient(),t)

        while True:
            # Uniform Sampling
            i = random.randint(0, n_samples-1)
            # Compute the component gradient
            g = self.problem.compute_component_gradient(i, False)
            # Gradient update
            self.problem.update(-self.eta/(t+1.0)*g)
            t += 1

            # Save the training results
            if t%n_samples == 0:
                self.compute_iteration_info(self.problem.compute_full_gradient(),t)

                # Check the stopping criterion
                if t > max_T or self.results['norm_grad'][-1] < self.epsilon:
                    break
    
    def svrg(self):
        n_samples = self.problem.get_n_sample()
        max_T = n_samples*self.max_pass
        t = 0
        
        while True:
            # Compute the full gradient
            g_old = self.problem.compute_full_gradient()

            # Save the training results
            self.compute_iteration_info(g_old,t)

            # Check the stopping criterion
            if t > max_T or self.results['norm_grad'][-1] < self.epsilon:
                break

            t += n_samples
            for s in xrange(self.epoch_size):
                # Uniform Sampling
                i = random.randint(0, n_samples-1)
                # Compute the svrg step
                g = self.problem.compute_component_gradient(i, False)
                - self.problem.compute_component_gradient(i, True) + g_old
                # Gradient update
                self.problem.update(-self.eta*g)
            t += self.epoch_size

    def set_epoch_size(self, _epoch_size_):
        self.epoch_size = _epoch_size_

    def compute_iteration_info(self, _gradient_, _curr_n_grad_comp_):
        if self.results_stored['trn_loss']:
            self.results['trn_loss'].append(self.problem.compute_loss(True))
        if self.results_stored['tst_loss']:
            self.results['tst_loss'].append(self.problem.compute_loss(False))
        if self.results_stored['trn_err']:
            y_predict_tmp, err_tmp = self.problem.compute_error(True)
            self.results['trn_err'].append(err_tmp)
        if self.results_stored['tst_err']:
            y_predict_tmp, err_tmp = self.problem.compute_error(False)
            self.results['tst_err'].append(err_tmp)
        if self.results_stored['norm_grad']:
            self.results['norm_grad'].append(np.linalg.norm(_gradient_))
        if self.results_stored['n_grad_comp']:
            self.results['n_grad_comp'].append(_curr_n_grad_comp_)


    def get_result(self, _key_):
        if _key_ not in self.results.keys():
            print "No such a key: [" +_key_ + "]!"
            exit()
        return self.results[_key_]        