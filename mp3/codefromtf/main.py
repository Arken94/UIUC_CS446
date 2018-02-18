"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *
from sklearn.metrics import accuracy_score, confusion_matrix

""" Hyperparameter for Training """
learn_rate = 0.00005
max_iters = 300


def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')

    # Initialize model.

    # Build TensorFlow training graph

    # Train model via gradient descent.

    # Compute classification accuracy based on the return of the "fit" method

    X, Y_true = read_dataset_tf('../data/trainset', 'indexing.txt')
    model = LogisticModel_TF(ndims=X.shape[1] - 1, W_init='zeros')
    model.build_graph(learn_rate=learn_rate)

    Y_reg = model.fit(Y_true, X, max_iters=max_iters).reshape(-1)
    Y_pred = np.array([1 if y_i > 0.5 else 0 for y_i in Y_reg])
    assert(Y_pred.shape == (X.shape[0],))
    assert(Y_true.shape == (X.shape[0], 1))
    print('Confusion matrix')
    print(confusion_matrix(Y_true.reshape(-1), Y_pred))
    print('Accuracy:', accuracy_score(Y_true.reshape(-1), Y_pred))


if __name__ == '__main__':
    tf.app.run()
