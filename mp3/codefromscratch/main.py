"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *
from sklearn.metrics import confusion_matrix


""" Hyperparameter for Training """
learn_rate = None
max_iters = None

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')

    # Initialize model.

    # Train model via gradient descent.

    # Save trained model to 'trained_weights.np'

    # Load trained model from 'trained_weights.np'

    # Try all other methods: forward, backward, classify, compute accuracy

    filename = 'trained_weights.np'

    x_train, y_train = read_dataset('../data/trainset', 'indexing.txt')
    model = LogisticModel(ndims=x_train.shape[1] - 1, W_init='zeros')
    model.fit(y_train, x_train, learn_rate=0.0001, max_iters=100)
    model.save_model(filename)

    model = LogisticModel(ndims=x_train.shape[1], W_init='zeros')
    model.load_model(filename)

    y_pred = model.classify(x_train)
    correct = np.count_nonzero(np.equal(y_pred, y_train))
    accuracy = 1. * correct / len(y_train)
    print('Accuracy:', accuracy)
    print(confusion_matrix(y_train, y_pred))
