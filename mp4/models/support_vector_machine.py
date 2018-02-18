"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        assert(f.shape == y.shape)
        assert(f.shape[1] == 1)

        reg_grad = self.w_decay_factor * self.w
        loss_grad = -(y * self.x).T.dot((y * f) < 1)
        total_grad = (reg_grad + loss_grad).reshape(-1)

        assert(total_grad.shape == (self.w.shape[0],))
        return total_grad

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor/2*||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """
        assert(f.shape == y.shape)
        assert(f.shape[1] == 1)
        hinge_loss = np.sum(np.maximum(np.zeros(f.shape), 1. - f * y))
        l2_loss = 0.5 * self.w_decay_factor * np.linalg.norm(self.w) ** 2.

        total_loss = hinge_loss + l2_loss
        # print('L', total_loss)
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,1). Tie break 0 to 1.0.
        """
        assert(f.shape[1] == 1)
        y_predict = np.array(
            [1. if f_i >= 0 else -1. for f_i in f.reshape(-1)]).reshape(f.shape)
        assert(y_predict.shape == f.shape)
        return y_predict
