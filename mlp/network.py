# -*- coding: utf-8 -*-

import time
import numpy as np
from .functions import *
from .data import shuffle


class network:
    """
    A simple multilayer perceptron implementation

    Attributes:
        n_layers (int): Number of layers

        n_neurons (list of int): Number of neurons in layers.
            len(n_neurons) = n_layers

        w (dict of np.ndarray): Weight matrices of the perceptron.
            shape: n_layers - 1, (n_input_neurons, n_output_neurons)
        
        b (dict of np.ndarray): Biases of the perceptron.
            shape: n_layers - 1, (n_neurons)

        z (dict of np.ndarray): Cache of intermediate z values
            for backpropagation. (input layer excluded)
            shape: n_layers - 1, (n_neurons)
        
        a (dict of np.ndarray): Cache of a values (layer activations)
            for backpropagation.
            shape: n_layers, (n_neurons)
        
        activations (list of activation): Activation function to use
            in layers (input layer excluded)
            len: n_layers - 1
        
        loss (cost): Loss function

    """

    def __init__(self, layers, activation_funcs):
        """
        Initialize a multilayer perceptron.

        Args:
            layers (list of int): List of number of neurons per layer
                ex: [4, 16, 1] - 4 input, 16 hidden, and 1 output
            
            activation_funcs (list of activation): Activation functions of each layer
                Length must be 2 (hidden, output) or n_layers - 1

        """

        self.n_layers = len(layers)
        self.n_neurons = layers

        # weights and biases
        self.w = {}
        self.b = {}
        for i in range(1, self.n_layers):
            # Xavier initialization: Gaussian with mean=0 and std=1/sqrt(n_input_neurons)
            self.w[i] = np.random.normal(loc=0.0,
                                         scale=1.0 / np.sqrt(self.n_neurons[i]),
                                         size=(self.n_neurons[i-1], self.n_neurons[i]))
            # self.w[i] = np.random.random(size=(self.n_neurons[i-1], self.n_neurons[i]))
            self.b[i] = np.zeros(shape=(self.n_neurons[i]))
        
        # intermediate values
        self.z = {}
        
        # activation values
        self.a = {}

        # functions used
        self.activations = {}
        if len(activation_funcs) == 2:
            for i in range(1, self.n_layers - 1):
                self.activations[i] = activation_funcs[0]
            self.activations[self.n_layers - 1] = activation_funcs[1]
        elif len(activation_funcs) == self.n_layers - 1:
            for i in range(1, self.n_layers):
                self.activations[i] = activation_funcs[i - 1]
        else:
            raise ValueError("Invalid number of activation functions")

        self.prev_avg_loss = float('inf')
    

    def _reset_w_b(self):
        """
        Resets the weights and biases

        """

        for i in range(1, self.n_layers):
            # Xavier initialization: Gaussian with mean=0 and std=1/sqrt(n_input_neurons)
            self.w[i] = np.random.normal(loc=0.0,
                                         scale=1.0 / np.sqrt(self.n_neurons[i]),
                                         size=(self.n_neurons[i-1], self.n_neurons[i]))
            # self.w[i] = np.random.random(size=(self.n_neurons[i-1], self.n_neurons[i]))
            self.b[i] = np.zeros(shape=(self.n_neurons[i]))


    def _forward(self, x):
        """
        Forward pass computes the z values and intermediate activation values
        a = f(z) = f(XW + b)

        Args:
            x (np.ndarray): Batch of samples. Each sample is a row vector, the shape of x
                is expected to be (batch_size, n_features). Batch size can be 1 in which
                case x will contain a single sample

        """

        self.a[0] = x
        for i in range(1, self.n_layers):
            self.z[i] = np.dot(self.a[i-1], self.w[i]) + self.b[i]
            self.a[i] = self.activations[i].forward(self.z[i])
        


    def predict(self, x):
        """
        Predict y value

        Args:
            x (np.ndarray): Input batch. Shape: (batch_size, n_neurons[0])

        Returns:
            y_pred (np.ndarray): Prediction for the given batch. Shape: (batch_size, n_neurons[-1])
        """

        self._forward(x)
        return self.a[self.n_layers - 1]


    def _update_params(self, update):
        """
        Apply the gradient to parameters

        Arguments:
            update (dict): Contains gradients and deltas

        """

        for index, (dw, delta) in update.items():
            self.w[index] -= self.lr * dw
            self.b[index] -= self.lr * np.mean(delta, axis=0)


    def _backward(self, y_true):
        """
        Backpropagation computes the gradient of cost function with respect to parameters using
        the cached values and updates the parameters

        """

        output_z = self.z[self.n_layers - 1]
        output_a = self.a[self.n_layers - 1]
        output_act = self.activations[self.n_layers - 1]

        # determine partial derivative and delta for the input of output activation function (output_z)
        delta = self.loss.delta(y_true, output_a)
        # delta = self.loss.backward(y_true, output_a) * output_act.backward(output_z, output_a)
        dw = np.dot(self.a[self.n_layers - 2].T, delta)

        update = {
            self.n_layers - 1: (dw, delta)
        }

        # each iteration requires the delta from the "higher" layer, propagating backwards.
        for i in reversed(range(1, self.n_layers - 1)):
            delta = np.dot(delta, self.w[i+1].T) * self.activations[i].backward(self.z[i], self.a[i])
            dw = np.dot(self.a[i-1].T, delta)
            update[i] = (dw, delta)

        self._update_params(update)


    def fit(self, x, y, loss, lr, n_epochs, batch_size, shuffle_data=True, val_ratio=None, val_data=None, es_epochs=None, es_delta=1e-5, metrics=None, print_stats=100):
        """
        Train the multilayer perceptron on the training set

        Args:
            x, y (np.ndarray): Training data

            loss (cost): Loss function to optimize
                default: categorical_crossentropy

            lr (float or tuple): Learning rate or annealing schedule in form (initial lr, every_n_epochs, multiplier)

            n_epochs (int): Number of epochs

            batch_size (int): Batch size

            shuffle_data (bool): Whether to shuffle data before each epoch
                default: True

            val_ratio (float): If provided, a fraction of data is used for validation
                default: None
            
            val_data (tuple of arrays): Data to be used as validation set
                default: None
            
            es_epochs (int): Number of epochs to watch the average loss for early stopping
                default: None
            
            es_delta (float): Early stopping tolerance
                default: 1e-5

            metrics (list of metric): Metric functions to compute every epoch
                default: None
            
            print_stats (int): Output stuff every _ epochs during training (silent if None)
                default: 1

        Returns:
            history (dict of dict of list): values of loss and metrics over epochs.
                Keys: 'metric_name': {'train', 'val'}

        """

        if not x.shape[0] == y.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        
        self.loss = loss
        
        if isinstance(lr, tuple):
            self.lr = lr[0]
            self.lr_n_epochs = lr[1]
            self.lr_multiplier = lr[2]
        elif isinstance(lr, float):
            self.lr = lr
            self.lr_n_epochs = -1
            self.lr_multiplier = 1
        else:
            raise TypeError("lr must be one of float or tuple")

        history = {
            'loss': {
                'train': [],
                'val': []
            }
        }

        if metrics:
            for m in metrics:
                m_name = m.name
                history[m_name] = {
                    'train': [],
                    'val': []
                }


        if val_ratio:
            idx = round(val_ratio * x.shape[0])
            x_train, y_train = x[idx:], y[idx:]
            x_val, y_val     = x[:idx], y[:idx]
            if print_stats:
                print(f"Train on {x_train.shape[0]} samples, validate on {x_val.shape[0]} samples for {n_epochs} epochs")
        elif val_data:
            x_train, y_train = x, y
            x_val, y_val = val_data[0], val_data[1]
            if print_stats:
                print(f"Train on {x_train.shape[0]} samples, validate on {x_val.shape[0]} samples for {n_epochs} epochs")
        else:
            x_train, y_train = x, y
            if print_stats:
                print(f"Train on {x_train.shape[0]} samples for {n_epochs} epochs")

        if print_stats:
            print("Stats are printed in format train / val")


        for e in range(n_epochs):

            t1 = time.perf_counter()

            # lr annealing
            if e != 0 and self.lr_n_epochs != -1 and e % self.lr_n_epochs == 0:
                self.lr *= self.lr_multiplier
                print(f"lr = {self.lr:.2e}")

            if shuffle_data:
                x_train, y_train = shuffle(x_train, y_train)

            for j in range(x_train.shape[0] // batch_size):
                k = j * batch_size
                l = min((j + 1) * batch_size, x_train.shape[0])
                self._forward(x_train[k:l])
                self._backward(y_train[k:l])

            # compute loss for the training set
            self._forward(x_train)
            train_loss = self.loss.forward(y_train, self.a[self.n_layers - 1])
            history['loss']['train'].append(train_loss)

            # compute metrics for the validation set
            for m in metrics:
                m_name = m.name
                # y_train_int = np.argmax(y_train, axis=1)
                y_train_int = y_train
                # y_train_pred_int = np.argmax(self.a[self.n_layers - 1], axis=1)
                y_train_pred_int = (self.a[self.n_layers - 1] > 0.5).astype(int)
                history[m_name]['train'].append(m(y_train_int, y_train_pred_int))

            if val_ratio or val_data:
                # compute loss for the validation set
                self._forward(x_val)
                val_loss = self.loss.forward(y_val, self.a[self.n_layers - 1])
                history['loss']['val'].append(val_loss)

                # compute metrics for the validation set
                for m in metrics:
                    m_name = m.name
                    # y_val_int = np.argmax(y_val, axis=1)
                    y_val_int = y_val
                    # y_val_pred_int = np.argmax(self.a[self.n_layers - 1], axis=1)
                    y_val_pred_int = (self.a[self.n_layers - 1] > 0.5).astype(int)
                    history[m_name]['val'].append(m(y_val_int, y_val_pred_int))

            millis = (time.perf_counter() - t1) * 1_000

            width = len(str(n_epochs))
            if print_stats and e % print_stats == 0:
                if val_ratio or val_data:
                    print(f"epoch {e:>{width}}/{n_epochs}:  loss: {train_loss:.4f} / {val_loss:.4f}  -  ", end='')
                else:
                    print(f"epoch {e:>{width}}/{n_epochs}:  loss: {train_loss:.4f}  -  ", end='')
                for m in metrics:
                    m_name = m.name
                    if val_ratio or val_data:
                        print(f"{m_name}: {history[m_name]['train'][-1]:.4f} / {history[m_name]['val'][-1]:.4f}  -  ", end='')
                    else:
                        print(f"{m_name}: {history[m_name]['train'][-1]:.4f}  -  ", end='')
                print(f"{millis:.2f} ms/epoch")
            
            if es_epochs and (val_ratio or val_data) and e >= es_epochs:
                avg_loss = np.mean(history['loss']['val'][-es_epochs:])
                if self.prev_avg_loss - avg_loss < 0.0001:
                    if print_stats:
                        print("early stopping")
                    return history
                self.prev_avg_loss = avg_loss

        return history


    def k_fold_cv(self, k, x, y, net_args):
        """
        K-fold cross-validation

        Args:
            net:         The network to test
            k:           K (default: 5)
            x, y:        The dataset
            net_args:    Network parameters
        
        Returns:
            res:   Dict of metrics (averaged)

        """

        scores = {}
        res = {}
        for metric in net_args['metrics']:
            scores[metric.name] = []
        x_, y_ = shuffle(x, y)

        for i in range(k):
            chunk_size = x_.shape[0] / k
            start_idx, end_idx = int(chunk_size * i), min(int(chunk_size * (i+1)), x_.shape[0])
            
            # validation
            x_val, y_val = x_[start_idx:end_idx], y_[start_idx:end_idx]
            
            # training
            x_t1, y_t1 = x_[:start_idx], y_[:start_idx]
            x_t2, y_t2 = x_[end_idx:], y_[end_idx:]
            x_train = np.concatenate([x_t1, x_t2])
            y_train = np.concatenate([y_t1, y_t2])

            # reset the model
            self._reset_w_b()

            # train the model
            _ = self.fit(x_train, y_train, val_data=(x_val, y_val), **net_args)
            
            # predict on test set
            y_val_pred = (self.predict(x_val) > 0.5).astype(int)

            for metric in net_args['metrics']:
                scores[metric.name].append(metric(y_val, y_val_pred))

        for metric in net_args['metrics']:
            res[metric.name] = np.mean(np.array(scores[metric.name]))

        return res