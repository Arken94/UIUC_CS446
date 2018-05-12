"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers
import pdb

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=2, learning_rate=0.0005):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims], name='x_placeholder')

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent], name='z_placeholder')

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        # y_hat = tf.sigmoid(self._discriminator(self.x_hat))
        # y = tf.sigmoid(self._discriminator(self.x_placeholder, reuse=True))
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

        # Add optimizers for appropriate variables
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.d_optimizer = optimizer.minimize(self.d_loss, var_list=d_vars)
        self.g_optimizer = optimizer.minimize(self.g_loss, var_list=g_vars)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1). 
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            layer0 = tf.layers.dense(x, 512, activation=tf.nn.relu)
            y = tf.layers.dense(layer0, 1)
            return y

    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        # return -tf.reduce_mean(tf.log(y) + tf.log(1. - y_hat))
        y_hat_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_hat), logits=1 - y_hat)
        y_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y), logits=y)
        l = tf.reduce_mean(y_hat_loss + y_loss)
        return l

    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation 
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            layer0 = tf.layers.dense(z, 128, activation=tf.nn.relu, name='layer0')
            x_hat = tf.layers.dense(layer0, self._ndims, activation=tf.nn.sigmoid)
            return x_hat

    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_hat), logits=y_hat))
        # l = -tf.reduce_mean(tf.log(y_hat))
        return l

    def train(self, x, z, learning_rate):
        self.session.run(self.g_optimizer,
                         feed_dict={self.z_placeholder: z})
        self.session.run(self.d_optimizer,
                         feed_dict={self.x_placeholder: x,
                                    self.z_placeholder: z})