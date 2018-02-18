"""logistic model class for binary classification."""

import numpy as np


class LogisticModel(object):
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term, 
            self.W = [Bias, W1, W2, W3, ...] 
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            self.W = np.zeros((ndims + 1, 1))
        elif W_init == 'ones':
            self.W = np.ones((ndims + 1, 1))
        elif W_init == 'uniform':
            self.W = np.random.rand((ndims + 1, 1))
        elif W_init == 'gaussian':
            self.W = np.random.randn((ndims + 1, 1))
        else:
            print('Unknown W_init ', W_init)
        
    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print ('model loaded from', weight_file)

    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return
                probability score (sigmoid).
        Args:
            X(numpy.ndarray):
                input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for each sample 
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        result = 1 / (1 + np.exp(-np.dot(X, self.W))).reshape(-1)
        assert(result.shape == (X.shape[0],))
        return result

    def backward(self, Y_true, X):
        """ Backward operation for logistic models. 
            Compute gradient according to the probability loss on lecture slides
        Args:
            X(numpy.ndarray):
                input dataset with a dimension of (# of samples, ndims+1)
            Y_true(numpy.ndarray):
                dataset labels with a dimension of (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        grad_num = -Y_true * X * np.exp(-Y_true * np.dot(X, self.W))
        grad_den = 1 + np.exp(-Y_true * np.dot(X, self.W))
        grad = grad_num / grad_den
        assert(grad.shape == (len(Y_true),))
        return grad

    def classify(self, X, threshold=0.5):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray):
                input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        Y_pred = self.forward(X)
        result = np.array([1 if y_i > threshold else -1 for y_i in Y_pred])
        assert(result.shape == (X.shape[0],))
        assert(result.shape == Y_pred.shape)
        return result

    def fit(self, Y_true, X, learn_rate, max_iters):
        """ train model with input dataset using gradient descent. 
        Args:
            Y_true(numpy.ndarray):
                dataset labels with a dimension of (# of samples,)
            X(numpy.ndarray):
                input dataset with a dimension of (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        assert(Y_true.shape == (len(Y_true),))
        assert(X.shape[1] == self.ndims + 1)
        for epoch in range(max_iters):
            Y_pred = np.matmul(X, self.W)
            grad = np.zeros((self.ndims + 1, 1))
            for i in range(len(Y_true)):
                grad_num = -Y_true[i] * X[i] * np.exp(-Y_true[i] * Y_pred[i])
                grad_den = 1 + np.exp(-Y_true[i] * Y_pred[i])
                grad += (grad_num / grad_den).reshape((self.ndims + 1, 1))
            self.W = self.W - learn_rate * grad
            # y_pred = self.classify(X)
            # correct = np.count_nonzero(np.equal(y_pred, Y_true))
            # accuracy = 1. * correct / len(Y_true)
            # print('L', self.loss(Y_true, X), 'acc', accuracy)

    def loss(self, Y_true, X):
        Y_pred = self.forward(X)
        loss = np.sum(np.log(1 + np.exp(-Y_true * Y_pred)))
        return loss
