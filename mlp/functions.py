# -*- coding: utf-8 -*-

"""
Activation, loss and cost functions.

"""

import numpy as np

_epsilon = 1e-10


class activation:
    """
    Abstract class of a differentiable function that
    contains a forward and a backward pass (for backprop)

    """

    def forward(self, z):
        """
        Calculates the output
        """
        raise NotImplementedError(f"{self.__class__.__name__}.forward() not implemented")

    def backward(self, z, a):
        """
        Calculates the gradient wrt z, (a: cached activation value)
        """
        raise NotImplementedError(f"{self.__class__.__name__}.backward() not implemented")


class cost:
    """
    Abstract class of a differentiable cost that
    contains a forward and a backward pass (for backprop)

    """

    def forward(self, y_true, y_pred):
        """
        Calculates the cost (scalar)
        """
        raise NotImplementedError(f"{self.__class__.__name__}.forward() not implemented")

    def backward(self, y_true, y_pred):
        """
        Calculates the gradient of cost wrt inputs
        """
        raise NotImplementedError(f"{self.__class__.__name__}.backward() not implemented")

    def delta(self, y_true, y_pred):
        """
        Calculates the gradient of cost wrt inputs of the output activation function
        """
        raise NotImplementedError(f"{self.__class__.__name__}.delta() not implemented")


# activation

class relu(activation):
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z, a):
        dz = np.ones_like(z)
        dz[dz < 0] = 0
        return dz


class leaky_relu(activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, z):
        y1 = ((z > 0) * z)
        y2 = ((z <= 0) * z * self.alpha)
        return y1 + y2

    def backward(self, z, a):
        dz = np.ones_like(z)
        dz[dz < 0] = self.alpha
        return dz


class tanh(activation):
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z, a):
        return 1 - a**2


class sigmoid(activation):
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, z, a):
        return a * (1 - a)


class linear(activation):
    def forward(self, z):
        return z

    def backward(self, z, a):
        return 1


class softmax(activation):
    def forward(self, z):
        def soft(x):
            x = x - np.max(x)
            row_sum = np.sum(np.exp(x))
            return np.array([np.exp(x_i) / row_sum for x_i in x])
        row_maxes = np.max(z, axis=1)
        row_maxes = row_maxes[:, np.newaxis]  # for broadcasting
        z = z - row_maxes
        return np.array([soft(row) for row in z])

    def backward(self, z, a):
        def grad(a):
            return np.diag(a) - np.outer(a, a)
        return np.array([grad(row) for row in a])


# cost

class mean_squared_error(cost):
    def forward(self, y_true, y_pred):
        return np.mean((y_pred - y_true)**2)

    def backward(self, y_true, y_pred):
        return y_pred - y_true


class categorical_crossentropy(cost):
    """
    Assumes that its input is softmaxed
    """
    def forward(self, y_true, y_pred):
        def cce(y_true, y_pred):
            # efficient, but assumes y is one-hot
            return -np.log(y_pred[np.where(y_true)])
        return np.mean([cce(y_row, s_row) for y_row, s_row in zip(y_true, y_pred)])

    def backward(self, y_true, y_pred):
        return -(1 / y_true.shape[0]) * (y_true / (y_pred + _epsilon))
    
    def delta(self, y_true, y_pred):
        """Gradient wrt input of output activation function"""
        return (1 / y_true.shape[0]) * (y_pred - y_true)


class binary_crossentropy(cost):
    """
    Assumes that its input is sigmoided
    """
    def forward(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        raise NotImplementedError("BCE backward not implemented")
 
    def delta(self, y_true, y_pred):
        """Gradient wrt input of output activation function"""
        return y_pred - y_true