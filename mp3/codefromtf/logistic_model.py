"""logistic model class for binary classification."""
import tensorflow as tf
import numpy as np


class LogisticModel_TF(object):
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of Weight is the bias term, 
            Weight = [Bias, W1, W2, W3, ...] 
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
        self.W0 = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            # Hint: self.W0 = tf.zeros([self.ndims+1,1])
            self.W0 = tf.zeros([ndims + 1, 1], dtype=tf.float64)
        elif W_init == 'ones':
            self.W0 = tf.ones([ndims + 1, 1], dtype=tf.float64)
        elif W_init == 'uniform':
            self.W0 = tf.random_uniform([ndims + 1, 1], dtype=tf.float64)
        elif W_init == 'gaussian':
            self.W0 = tf.random_normal([ndims + 1, 1], dtype=tf.float64)
        else:
            print ('Unknown W_init ', W_init) 

    def build_graph(self, learn_rate):
        """ build tensorflow training graph for logistic model.
        Args:
            learn_rate: learn rate for gradient descent
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # Hint: self.W = tf.Variable(self.W0)
        self.X = tf.placeholder(tf.float64, name='X',
                                shape=(None, self.ndims + 1))
        self.Y = tf.placeholder(tf.float64, name='Y', shape=(None,1))
        self.W = tf.Variable(self.W0, dtype=tf.float64)

        # g = 1 / (1 + tf.exp(-tf.matmul(self.X, self.W)))
        g = tf.sigmoid(tf.matmul(self.X, self.W))
        loss = tf.reduce_sum(tf.pow(self.Y - g, 2.))
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learn_rate).minimize(loss)

    def fit(self, Y_true, X, max_iters):
        """ train model with input dataset using gradient descent. 
        Args:
            Y_true(numpy.ndarray):
                dataset labels with a dimension of (# of samples,1)
            X(numpy.ndarray):
                input dataset with a dimension of (# of samples, ndims+1)
            max_iters: maximal number of training iterations
            ......: append as many arguments as you want
        Returns:
            (numpy.ndarray): sigmoid output from well trained logistic model,
                             used for classification with a dimension
                             of (# of samples, 1)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(max_iters):
                sess.run(self.optimizer, feed_dict={self.X: X, self.Y: Y_true})
                # result = sess.run(
                #     tf.reduce_sum(
                #         tf.pow(Y_true-tf.sigmoid(tf.matmul(X, self.W)), 2.)))
                # print('Epoch', epoch, 'L', result)

            sigmoid = sess.run(1 / (1 + tf.exp(-tf.matmul(X, self.W))))
            assert(sigmoid.shape == (len(Y_true), 1))
            return sigmoid
