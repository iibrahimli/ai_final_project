import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dtree
from mlp import data, metrics

np.random.seed(25)


# import data
df = pd.read_csv("datasets/heart_disease.csv", sep=';')
print(f"Data contains {(df['target'].values == 1).sum()} positives ({(df['target'].values == 1).sum() / len(df) * 100:.2f} %)")
print(f"Number of features in the dataset: {df.shape[1]}")


"""
    Decision trees do not require much numerical preprocessing, since
    their performance does not directly depend on the range and mean
    of the data. The only manipulation done on the data is the discretization
    of continuous features (done in dtree class)
"""

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values[..., np.newaxis]

# prepare training and test sets
x, y = data.shuffle(x, y)
(x_train, y_train), (x_test, y_test) = data.split(x, y, ratio=0.8)

print()
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test:",  x_test.shape,  "y_test:",  y_test.shape)
print()


# create (and train) the tree

n_groups = 3

dt = dtree.dtree(
    (x_train, y_train),
    n_groups=n_groups,
    max_depth=4
)

print(f"Train on {len(x_train)} samples, validate on {len(x_test)} samples, n_groups: {n_groups}")


# evaluate on test set
y_test_pred = dt.predict(x_test)

# calculate metrics
tp, tn, fp, fn = metrics.tfpn(y_test, y_test_pred)
test_acc  = metrics.accuracy()(y_test, y_test_pred)
test_prec = metrics.precision()(y_test, y_test_pred)
test_sens = metrics.sensitivity()(y_test, y_test_pred)
test_spec = metrics.specificity()(y_test, y_test_pred)
test_f1   = metrics.f1()(y_test, y_test_pred)

print()
print("Evaluation on test set:")
print("----------")
print(f"TP: {tp}")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"accuracy:    { test_acc:.3f}")
print(f"precision:   {test_prec:.3f}")
print(f"sensitivity: {test_sens:.3f}")
print(f"specificity: {test_spec:.3f}")
print(f"f1:          {test_f1:.3f}")