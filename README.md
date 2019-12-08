# L3 Artificial Intelligence Final Project

Everything is written in Python 3, using Pandas and NumPy.

## Installation

Start by cloning this repository:
```
$ git clone https://github.com/iibrahimli/ai_final_project.git
```
and `cd` into the project directory
```
$ cd ai_final_project
```

## Usage

There are 2 libraries (*mlp* and *dtree*) and 2 main files (`main_mlp.py` and `main_dtree.py`).

### MLP

To test the MLP:
```
$ python3 main_mlp.py
```
To plot the loss and metrics during training change False to True in the last section of the code. Other network parameters can be tuned too. Custom activation, loss, and metric functions can be implemented in the corresponding `.py` file (don't forget to inherit from the provided base classes).

### Decision Tree

To test the decision tree, run
```
$ python3 main_dtree.py
```

Parameters like tree depth, n_groups may be tuned.

---

### Authors
| Imran Ibrahimli | Sanan Najafov | Sabina Hajimuradova |
|:-:|:-:|:-:|