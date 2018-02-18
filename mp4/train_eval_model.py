"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    ids = np.arange(data['image'].shape[0])
    n_batches = np.ceil(1. * len(ids) / batch_size)
    done = False
    iter_num = 0
    while not done:
        dataset = data
        if shuffle:
            np.random.shuffle(ids)
            dataset = {
                'image': data['image'][ids],
                'label': data['label'][ids]
            }
        x_batches = np.array_split(dataset['image'], n_batches)
        y_batches = np.array_split(dataset['label'], n_batches)
        for x_batch, y_batch in zip(x_batches, y_batches):
            update_step(x_batch, y_batch, model, learning_rate)
            iter_num += 1
            if iter_num == num_steps:
                done = True
                break
        # model.total_loss(model.forward(dataset['image']), dataset['label'])
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
        learning_rate(float): hyper-parameter
    """
    f = model.forward(x_batch)
    gradient = model.backward(f, y_batch).reshape(model.ndims + 1, 1)
    model.w = model.w - learning_rate * gradient


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)

    # Set model.w
    model.w = z[:model.ndims + 1]


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    C = model.w_decay_factor
    N = data['image'].shape[0]
    D = model.ndims + 1
    y = data['label']
    x = np.concatenate((data['image'], np.ones((N, 1))), axis=1)

    id_N = np.identity(N)
    y_constraints = -np.concatenate((y * x, id_N), axis=1)
    xi_constraints = np.concatenate((np.zeros(x.shape), -id_N), axis=1)

    padded_id = np.zeros((N + D, N + D))
    padded_id[:D,:D] = np.identity(D)

    P = C * padded_id
    q = np.concatenate((np.zeros((D, 1)), np.ones((N, 1))))
    G = np.concatenate((y_constraints, xi_constraints))
    h = np.concatenate((-np.ones((N, 1)), np.zeros((N, 1))))

    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    x_eval, y_eval = data['image'], data['label']
    f = model.forward(x_eval)
    y_pred = model.predict(f)
    loss = model.total_loss(f, y_eval)
    acc = 1. * np.count_nonzero(y_eval * y_pred == 1) / len(y_pred)
    return loss, acc
