import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlp import data, functions, metrics, network

np.random.seed(25)
show_plots = False


# importing the dataset
"""
Chapter 1
    1. We want to predict the attribute 'target'
    2. Binary classification, since there are 2 possible cases (healthy or not)
    3. 'sex', 'chest_pain_type', 'fasting_blood_sugar', 'rect_ecg',
       'exercise_induced_angina', 'st_slope', 'num_major_vessels', 'thalessemia',
       and 'target'
    4. Using [multilabel] one-hot encoding
"""

df = pd.read_csv("datasets/heart_disease.csv", sep=';')
# print(df.info())
# print(df.head())
print(f"Data contains {(df['target'].values == 1).sum()} positives ({(df['target'].values == 1).sum() / len(df) * 100:.2f} %)")
print(f"Number of features in the dataset: {df.shape[1]}")


# preprocessing


# convert categorical features to one hot
df, means, stds = data.preprocess_df(df, [
    'sex',
    'chest_pain_type',
    'fasting_blood_sugar',
    'rest_ecg',
    'exercise_induced_angina',
    'st_slope',
    'num_major_vessels',
    'thalassemia'
])


# get features
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values[..., np.newaxis]
print(f"Number of features after one-hot encoding: {x.shape[1]}")

# prepare training and test sets
x, y = data.shuffle(x, y)
(x_train, y_train), (x_test, y_test) = data.split(x, y, ratio=0.7)

print()
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test:",  x_test.shape,  "y_test:",  y_test.shape)
print()


# the model
"""
Chapter 2
    1. There are 2 weight matrices with shapes (31, 5) and (5, 1).
       The bias vectors have shapes (5,) and (1,).
       Mini-batches can be elegantly processed by defining the weight
       matrices appropriately so that they can be used in form
                        a = f(XW + b)
       where X is the input matrix formed by stacking input features (row vectors)
    2. We stop the training when the validation loss stops decreasing
    3. See report/network_arch TODO
"""

net = network.network(
    [x_train.shape[1], 5, 1],
    [
        functions.relu(),
        functions.sigmoid()
    ]
)


# training
history = net.fit(
    x_train, y_train,
    loss=functions.binary_crossentropy(),
    lr=(0.1, 100, 0.5),                  # learning rate annealing
    n_epochs=500,
    batch_size=4,
    val_data=(x_test, y_test),
    es_epochs=10,
    es_delta=1e-8,
    metrics=[
        metrics.accuracy(),
        metrics.sensitivity(),
        metrics.specificity(),
        metrics.f1()
    ],
    print_stats=1
)


# evaluation
"""
    * In the case of predicting the risk of heart disease, sensitivity would be more
      important than specificity, because the patients who have been falsely identified
      as healthy can neglect their heart condition, possibly leading to death. Thus
      it is better to have the minimal number of 
"""

# convert probabilities to integer labels
y_test_hat = net.predict(x_test)
y_test_pred = (y_test_hat > 0.5).astype(int)


# for y_true, y_hat in zip(y_test, y_test_hat):
    # print(f"{y_true[0]:.4f}  -  {y_hat[0]:.4f}")
# exit(0)


# calculate metrics
test_acc  = metrics.accuracy()(y_test, y_test_pred)
test_prec = metrics.precision()(y_test, y_test_pred)
test_sens = metrics.sensitivity()(y_test, y_test_pred)
test_spec = metrics.specificity()(y_test, y_test_pred)
test_f1   = metrics.f1()(y_test, y_test_pred)

print()
print("----------")
print("Evaluation results on test set:")
print(f"accuracy:    { test_acc:.3f}")
print(f"precision:   {test_prec:.3f}")
print(f"sensitivity: {test_sens:.3f}")
print(f"specificity: {test_spec:.3f}")
print(f"f1:          {test_f1:.3f}")


# plot training stats
if show_plots:    
    n_plots = len(history)
    fig, axs = plt.subplots(1, n_plots)
    for i, m in enumerate(history.keys()):
        axs[i].plot(history[m]['train'], label=f'train_{m}')
        axs[i].plot(history[m]['val'], label=f'val_{m}')
        axs[i].legend()
        axs[i].set_xlabel('epochs')
        axs[i].set_ylabel(m)
    plt.tight_layout()
    plt.show()