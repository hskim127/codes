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

    # Full gradient descent start
    def full_gradient_descent(self):
        # Local copy
        _n_samples_ = self.problem.get_n_sample()
        _max_T_ = _n_samples_*self.max_pass
        _problem_ = self.problem
        _epsilon_ = self.epsilon
        _eta_ = self.eta

        t = 0
        while True:
            # Compute the component gradient
            g = _problem_.compute_full_gradient()

            # Save the training results
            self.compute_iteration_info(g,t)

            # Check the stopping criterion
            if t > _max_T_ or self.results['norm_grad'][-1] < _epsilon_:
                break

            # Gradient update
            _problem_.update(-_eta_*g)
            t += _n_samples_
        self.problem = _problem_
    # Full gradient descent end

    # SGD start : It may not work. I don't use this function since I just compare SVRG w/ GD, not SGD.
    def stochastic_gradient_descent(self):
        # Local copy
        _n_samples_ = self.problem.get_n_sample()
        _max_T_ = _n_samples_*self.max_pass
        _problem_ = self.problem
        _epsilon_ = self.epsilon
        _eta_ = self.eta

        t = 0
        while True:
            # Uniform Sampling
            i = random.randint(0, _n_samples_-1)
            # Compute the component gradient
            g = _problem_.compute_component_gradient(i, False)

            # Save the training results
            if t%_n_samples_ == 0:
                self.compute_iteration_info(g,t)

                # Check the stopping criterion
                if t > _max_T_ or self.results['norm_grad'][-1] < _epsilon_:
                    break

            # Gradient update
            _problem_.update(-_eta_/(t+1.0)*g)
            t += 1
        self.problem = _problem_
    # SGD end

    # SVRG start
    def svrg(self):
        # Local copy
        _n_samples_ = self.problem.get_n_sample()
        _max_T_ = _n_samples_*self.max_pass
        _problem_ = self.problem
        _epoch_size_ = self.epoch_size
        _epsilon_ = self.epsilon
        _eta_init_ = self.eta
        _eta_ = self.eta

        alpha = 0.9
        t = 0
        t_tmp = 0 # To match the number of computations of Full gradient algorithm
        while True:
            # Compute the full gradient
            g_old = _problem_.compute_full_gradient()

            # Save the training results
            self.compute_iteration_info(g_old,t)

            # Check the stopping criterion
            if t_tmp > _max_T_ or self.results['norm_grad'][-1] < _epsilon_:
                break

            # Uniform Sampling w/ replacement
            ii = np.random.randint(low=0,high=_n_samples_,size=_epoch_size_)
            # In the PIM setting, we don't count this computation.
            t += _n_samples_
            for i in ii:
                # Compute the svrg step
                g = _problem_.compute_component_gradient(i, False) - _problem_.compute_component_gradient(i, True) + g_old

                # Step-size rescaling
                _eta_ = alpha*_eta_ if _eta_ > 1e-4 else _eta_init_

                # Gradient update
                _problem_.update(-_eta_*np.array(g.tolist()[0]))

            t += _epoch_size_
            t_tmp += _epoch_size_
        self.problem = _problem_
    # SVRG end

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