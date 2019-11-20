# -*- coding: utf-8 -*-

"""
Function for data preprocessing
"""

import numpy as np
import pandas as pd


def feat_normalize(x):
    """
    Column-wise min-max normalize an array of features to range [0, 1]

    Args:
        x (np.ndarray): Feature array to normalize
    
    Returns:
        res (np.ndarray):  Result
        mins (np.ndarray): Feature minimums
        maxs (np.ndarray): Feature maximums
    """

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    res = (x - mins) / (maxs - mins)
    return res, mins, maxs


def feat_standardize(x):
    """
    Column-wise standardize an array of features to have
    mean=0 and std=1

    Args:
        x (np.ndarray): Feature array to standardize
    
    Returns:
        res (np.ndarray):   Result
        means (np.ndarray): Feature means
        stds (np.ndarray):  Feature standard deviations
    """

    means = x.mean(axis=0)
    stds = x.std(axis=0)
    res = (x - means) / stds
    return res, means, stds


def df_standardize(df, numerical_columns):
    """
    Standardize the columns of features to have
    mean=0 and std=1

    Args:
        df (pd.DataFrame): Dataframe to standardize
        num_col (list):    Names of numerical columns
    
    Returns:
        res (np.ndarray):  Result
        means (dict):      Feature means
        stds (dict):       Feature standard deviations
    """

    means = {}
    stds = {}
    res = df.copy()
    for col in numerical_columns:
        mu = res[col].mean()
        sigma = res[col].std()
        means[col] = mu
        stds[col] = sigma
        res[col] = (res[col] - mu) / sigma
    return res, means, stds


def arr_to_one_hot(x):
    """
    One-hot encode integer labels

    Args:
        x (np.ndarray): Array of integer labels
        Shape: (n_samples,)

    Returns:
        oh (np.ndarray): Array of one hot encoded labels
        Shape: (n_samples, n_classes)
    """

    oh = np.zeros((x.size, x.max()+1))
    oh[np.arange(x.size), x] = 1
    return oh


def shuffle(x, y):
    """
    Randomly shuffle data

    Args:
        x, y (np.ndarray): Data
    
    Returns:
        x, y (np.ndarray): Shuffled data
    """

    seed = np.arange(x.shape[0])
    np.random.shuffle(seed)
    return x[seed], y[seed]


def split(x, y, ratio=0.7):
    """
    Split data into training and testing sets
    
    Args:
        x, y (np.ndarray): Data
        ratio (float):     Fraction of training data
    
    Returns:
        (x_train, y_train), (x_test, y_test): Data after splitting
    """

    idx = round(len(x) * ratio)
    x_train, y_train = x[:idx], y[:idx]
    x_test,  y_test  = x[idx:], y[idx:]
    return (x_train, y_train), (x_test, y_test)


def to_one_hot(df, columns):
    """
    One-hot encode given columns

    Args:
        df (pd.DataFrame):      Dataframe to encode
        columns (list of str):  List of column names to encode
    
    Returns:
        enc_df (pd.DataFrame):  Encoded dataframe
    """

    return pd.get_dummies(df, columns=columns, prefix=columns)


def preprocess_df(df, categorical_columns):
    """
    One-hot encode categorical columns and normalize other columns

    Args:
        df (pd.DataFrame):      Dataframe to encode
        columns (list of str):  List of column names to encode
            All of the other columns are considered numerical 
    
    Returns:
        enc_df (pd.DataFrame):  Encoded dataframe
    
    """

    num_cols = [col for col in df.columns if col not in categorical_columns]
    res, means, stds = df_standardize(df, num_cols)
    res = to_one_hot(res, categorical_columns)
    return res, means, stds