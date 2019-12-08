# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class dtree():

    class __rule():
    
        def __init__(self, col, val):
            """
            A rule of a node of decision tree

            Args:
                col: column that defines the rule
                val: value that defines the rule

            """

            self.__col = col
            self.__val = val

        
        def is_satisfied(self, data):
            """
                Checks whether rule is satisfied given a data point

                Args:
                    data: data point of shape 1xm

            Returns:
                bool - whether rule is satisfied or not

            """
            if self.__col == None and self.__val == None:
                return True
            
            return data[0, self.__col] == self.__val
        
        def __str__(self):
            """
                Returns rule as a string

            """

            return f"col: {self.__col}, val: {self.__val}"


    def __init__(self, data, n_groups=3, max_depth=5, indexes="all", cols_exclude=[], rule=None, groups=[]):
        """
        Decision Tree

        Args:
            data:        tupple containing x of shape nxm and y of shape kxm
            n_groups:    number of groups to get after discretization of the input data
            max_depth:   maximum depth of the tree
            indexes:     indexes of the data that are associated with the decision tree. Either "all" or a numpy array of shape n
            cols_exlude: columns that are excluded
            rule:        rule that defines the decision tree
            groups:      maximum values of the data points to be included in the ith group

        """

        self.__rule_        = rule
        self.__max_depth    = max_depth
        self.__data         = data
        self.__branches     = []
        self.__cols_exclude = cols_exclude
        self.__n_groups     = n_groups
        self.__groups       = groups

        if isinstance(indexes, str) and indexes == "all":
            self.__indexes = np.ones(len(data[0]), dtype=bool)
        else:
            self.__indexes = indexes.copy()

        self.float_to_one_hot()
        
        if max_depth > 0:
            self.__branches = self.__branch()

    
    def get_rule(self):
        """
        Returns the rule that defines the decision tree

        """

        return self.__rule_

    
    def get_data(self):
        return self.__data

    
    def get_groups(self):
        return self.__groups


    def float_to_one_hot(self, data=[]):
        """
        Encodes float data as one-hot

        Args:
            data: numpy array of shape nxm 

        Returns:
            one-hot encoded numpy array

        """

        if len(data) == 0:
            if len(self.__groups) != 0:
                return self.__data[0], self.__groups
            data = self.__data[0].copy()
            groups = np.zeros((self.__data[0].shape[1], self.__n_groups))

            for col in range(data.shape[1]):
                data = data[data[:, col].argsort()]
                for i in range(self.__n_groups):
                    cur_val = data[(i + 1) * data.shape[0] // self.__n_groups - 1, col]
                    groups[col, i] = cur_val

            new_d = np.empty(data.shape)
            
            for col in range(groups.shape[0]):
                for i in range(groups.shape[1]):
                    if i == 0:
                        new_d[:, col][self.__data[0][:, col] <= groups[col, i]] = i
                    else:
                        new_d[:, col][(self.__data[0][:, col] <= groups[col, i]) & (self.__data[0][:, col] > groups[col, i - 1])] = i
                new_d[:, col][self.__data[0][:, col] > groups[col, -1]] = self.__n_groups - 1

            self.__groups = groups.copy()
            self.__data = (new_d.copy(), self.__data[1])

        else:
            groups = self.__groups
            new_d = np.empty(data.shape)
            for col in range(groups.shape[0]):
                for i in range(groups.shape[1]):
                    if i == 0:
                        new_d[:, col][data[:, col] <= groups[col, i]] = i
                    else:
                        new_d[:, col][(data[:, col] <= groups[col, i]) & (data[:, col] > groups[col, i - 1])] = i
                new_d[:, col][data[:, col] > groups[col, -1]] = self.__n_groups - 1

        return new_d, groups


    def __branch(self):
        """
        Creates a new branch based on discriminative power of columns

        Returns:
            new decision tree

        """

        if len(np.unique(self.__data[1][self.__indexes])) <= 1:
            return []

        branches     = []
        disc_max     = -np.inf
        disc_max_col = None

        for col in range(self.__data[0].shape[1]):
            if col in self.__cols_exclude:
                continue
            disc = self.disc(col)
            if disc > disc_max:
                disc_max     = disc
                disc_max_col = col

        if disc_max_col == None:
            return branches
            
        uniques = np.unique(self.__data[0][self.__indexes, disc_max_col])
        cols_exclude = [col for col in self.__cols_exclude]
        cols_exclude.append(disc_max_col)
        for unique in uniques:
            indexes = (self.__data[0][:, disc_max_col] == unique)
            indexes = np.logical_and(self.__indexes, indexes)
            rule = self.__rule(disc_max_col, unique)
            branches.append(dtree(self.__data, self.__n_groups, self.__max_depth - 1, indexes, cols_exclude, rule, self.__groups))
            
        return branches
    

    def entropy(self, col=None):
        """
        Calculates entropy of a dataset based on column and value or the dataset itself

        Args:
            col: column to determine the entropy of. Calculates entropy of the dataset by default

        Returns:
            entropy as float

        """

        res = 0
        if col == None:
            uniques, counts = np.unique(self.__data[1][self.__indexes], return_counts=True)
            for val, count in zip(uniques, counts):
                if len(self.__data[1][self.__indexes] == val) != 0:
                    res -= (count / len(self.__data[1][self.__indexes]) * np.log2(count / len(self.__data[1][self.__indexes])))
        else:
            uniques_y = np.unique(self.__data[1][self.__indexes], return_counts=True)
            uniques_x, counts_x = np.unique(self.__data[0][self.__indexes, col], return_counts=True)
            for unique_x, count_x in zip(uniques_x, counts_x):
                y_indexes = np.logical_and(self.__indexes, (self.__data[0][:, col].reshape(-1) == [unique_x]))
                e = 0
                for unique_y in uniques_y:
                    y = self.__data[1][np.logical_and(y_indexes, (self.__data[1].reshape(-1) == [unique_y]))]
                    if len(y) != 0:
                        e += len(y) / count_x * np.log2(len(y) / count_x) 

                res -= (count_x / len(self.__indexes) * e)
        return res

    
    def disc(self, col):
        """
        Calculates discriminative power of the variable

        Args:
            col: the column to calculate the discriminative power of

        Returns:
            discriminative power as float

        """

        res = self.entropy() - self.entropy(col)

        return res

    
    def __predict(self, data_x):
        """
        Predicts the class of a data point

        Args:
            data_x: a data point of shape 1xm

        Returns:
            name of class

        """

        if len(self.__branches) == 0:
            vals, counts = np.unique(self.__data[1][self.__indexes], return_counts=True)
            return vals[np.argmax(counts)]
        else:
            corresponding_branch = None
            for branch in self.__branches:
                if branch.get_rule().is_satisfied(data_x):
                    corresponding_branch = branch
                    break
            if corresponding_branch == None:
                # No such data point was detected before
                unique, counts = np.unique(self.__data[1], return_counts=True)
                return unique[np.argmax(counts)]
            return corresponding_branch.__predict(data_x)

    
    def predict(self, data_x):
        """
        Predicts the class of data

        Args:
            data_x: data of shape nxm

        Returns:
            numpy array of shape nx1 containing classes

        """

        results = np.empty((data_x.shape[0], 1), dtype=self.__data[1].dtype)
        data = data_x.copy()

        data, _ = self.float_to_one_hot(data)

        for i, dp in enumerate(data):
            results[i, 0] = self.__predict(dp.reshape(1, -1))
        
        return results