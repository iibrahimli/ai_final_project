# -*- coding: utf-8 -*-

"""
Performance metrics.

"""

import numpy as np


def tfpn(y_true, y_pred):
    """
    Helper function to calculate TP, TN, FP, FN

    Args:
        y_true (np.ndarray): Ground truth value
        y_pred (np.ndarray): Predicted value
    
    Returns:
        tp, tn, fp, fn:      Values
    """

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


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
        tp, tn, fp, fn = tfpn(y_true, y_pred)
        return tp / (tp + fp)


class sensitivity(metric):
    def __call__(self, y_true, y_pred):
        tp, tn, fp, fn = tfpn(y_true, y_pred)
        return tp / (tp + fn)


class specificity(metric):
    def __call__(self, y_true, y_pred):
        tp, tn, fp, fn = tfpn(y_true, y_pred)
        return tn / (tn + fp)


class f1(metric):
    def __call__(self, y_true, y_pred):
        tp, tn, fp, fn = tfpn(y_true, y_pred)
        return 2 * tp / (2 * tp + fp + fn)


class kappa(metric):
    def __call__(self, y_true, y_pred):
        pass


def k_fold_cv(net, k, x, y):
    """
    K-fold cross-validation

    Args:
        net:   The network to test
        k:     K (default: 5)
        x, y:  The dataset
    
    Returns:
        res:   Dict of metrics (averaged)

    """

    # TODO
    pass