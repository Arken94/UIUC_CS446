"""Generative Adversarial Networks
"""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan
from time import time

plt.ion()

def train(model, mnist_dataset, learning_rate=1e-3, batch_size=16,
          num_steps=5000):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.uniform(-1, 1, [batch_size, model._nlatent])
        # Train generator and discriminator
        model.train(batch_x, batch_z, learning_rate=learning_rate)

        if step % 500 == 0:
            out = np.empty((28 * 20, 28 * 20))
            for x_idx in range(20):
                for y_idx in range(20):
                    z = np.random.uniform(-1, 1, [1, model._nlatent])
                    img = model.session.run(model.x_hat, feed_dict={model.z_placeholder: z})
                    out[x_idx * 28:(x_idx + 1) * 28,
                    y_idx * 28:(y_idx + 1) * 28] = img[0].reshape(28, 28)
            plt.imshow(out, cmap="gray")
            plt.pause(1e-9)
            plt.show()


def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan(nlatent=10, learning_rate=1e-3)

    # Start training
    train(model, mnist_dataset, num_steps=20000)


if __name__ == "__main__":
    tf.app.run()
