"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    ids = np.arange(processed_dataset[0].shape[0])
    n_batches = np.ceil(1. * len(ids) / batch_size)
    done = False
    iter = 0
    while not done:
        dataset = processed_dataset
        if shuffle:
            np.random.shuffle(ids)
            dataset = [processed_dataset[0][ids], processed_dataset[1][ids]]
        x_batches = np.array_split(dataset[0], n_batches)
        y_batches = np.array_split(dataset[1], n_batches)
        for x_batch, y_batch in zip(x_batches, y_batches):
            update_step(x_batch, y_batch, model, learning_rate)
            iter += 1
            if iter == num_steps:
                done = True
                break
        # model.total_loss(model.forward(dataset[0]), dataset[1])
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(x_batch)
    gradient = model.backward(f, y_batch)
    model.w = model.w - learning_rate * gradient


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    x_train, y_train = processed_dataset[0], processed_dataset[1]
    # add column of 1 for bias term
    N = x_train.shape[0]
    x_train = np.column_stack((x_train, np.ones((N, 1))))
    x_inner = x_train.T.dot(x_train)
    model.w = np.linalg.inv(x_inner + model.w_decay_factor * np.identity(x_inner.shape[0])).dot(x_train.T.dot(y_train))


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    x_eval, y_eval = processed_dataset[0], processed_dataset[1]
    f = model.forward(x_eval)
    loss = model.total_loss(f, y_eval)

    return loss
