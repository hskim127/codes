import numpy as np
import scipy as sp
import scipy.io as sio
from numpy import linalg as LA
from scipy.sparse import linalg as SLA

class BinaryLogisticRegression:
    def __init__(self,data,parameters):
        # Assumed that the data is given in the form of "Compressed Sparse Row matrix" of scipy or numpy array.
        self.x_trn = data['X_train']
        self.y_trn = data['y_train']

        # The number of training/test samples
        self.n_trn_samples = self.x_trn.shape[0]

        # Feature dimension
        self.dim = self.x_trn.shape[1]

        if 'X_test' in data.keys():
            self.x_tst = data['X_test']
            self.y_tst = data['y_test']
            self.n_tst_samples = self.x_tst.shape[0]
            if self.x_trn.shape[1] != self.x_tst.shape[1]:
                print "Dimensionality not matched beween the training and test data!"
                exit()

        # l2-regulariztion constant
        self.lam = parameters["lambda"]

        # Parameter to be trained (maybe need to have some other initialization options..?)
        self.w = np.zeros(self.dim)

        # Variable to store coefficients: recall that the gradient of f(x^w) = f'(x^w)*x
        self.grad_coeff = np.zeros(self.n_trn_samples)


    def compute_loss(self,is_training=True):
        n_sample = self.get_n_sample(is_training)
        if is_training:
            X, y = self.x_trn, np.array(self.y_trn)
        else:
            X, y = self.x_tst, np.array(self.y_tst)
        w = self.w
        
        loss = np.sum(np.log(1+np.exp(-y*X.dot(w))))
        loss /= float(n_sample)
        if self.lam != 0:
            loss += 0.5*self.lam*LA.norm(w,2)**2
        return loss

    def compute_full_gradient(self):
        X, y = self.x_trn, np.array(self.y_trn)
        n_sample = self.n_trn_samples
        w = self.w

        self.grad_coeff = -1/(1+np.exp(y*X.dot(w)))*y
        self.full_grad = X.transpose().dot(self.grad_coeff)
        self.full_grad /= float(n_sample)
        
        if self.lam != 0:
            self.full_grad += self.lam*w
        return self.full_grad

    def compute_component_gradient(self, sample_idx, is_old=False):
        x, y = self.x_trn[sample_idx,:], self.y_trn[sample_idx]
        w = self.w
        
        self.component_grad = self.grad_coeff[sample_idx]*x if is_old else -y/(1+np.exp(y*x.dot(w)))*x
        if self.lam != 0:
            self.full_grad += self.lam*w

        return self.component_grad
    
    def update(self, update_step):
        # 여기부터... 왜 갑자기 지랄이여 =_=.. i sampling 하는 걸 np.random.rand_int 로 array로 하고, 각 SGD iteration 마다 array로 접근하게 만든 후에 이렇게 됨
        self.w += update_step
        
    def get_n_sample(self,is_training=True):
        return self.n_trn_samples if is_training else self.n_tst_samples

    def predict(self,is_training=True):
        X = self.x_trn if is_training else self.x_tst
        y = np.ones(X.shape[0])
        y[X.dot(self.w)<0] = -1.0
        return y.tolist()

    def compute_error(self,is_training=True):
        y = np.array(self.y_trn if is_training else self.y_tst)
        y_predict = np.array(self.predict(is_training))
        err = np.mean(y != y_predict)
        return y_predict, err

    def get_model(self):
        return self.w