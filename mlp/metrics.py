# -*- coding: utf-8 -*-

"""
Performance metrics.

"""

import numpy as np


class metric:
    """
    Abstract class of a metric function that returns a scalar given predicted
    and true output values
    """

    def __call__(self, y_true, y_pred):
        """
        Calculates the output

        Args:
            y_true (np.ndarray): Ground truth value
            y_pred (np.ndarray): Predicted value
        
        Returns:
            res: Scalar metric value
        """

        raise NotImplementedError(f"{self.__class__.__name__}.__call__() not implemented")


class accuracy(metric):
    def __call__(self, y_true, y_pred):
        return np.sum(y_pred == y_true) / y_true.shape[0]


class precision(metric):
    def __call__(self, y_true, y_pred):
        pass


class sensitivity(metric):
    def __call__(self, y_true, y_pred):
        pass


class sensitivity(metric):
    def __call__(self, y_true, y_pred):
        pass


class f1(metric):
    def __call__(self, y_true, y_pred):
        pass


class kappa(metric):
    def __call__(self, y_true, y_pred):
        pass