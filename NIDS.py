#!/usr/bin/env python3

"""NIDS.py: Analyzes collected network traffic and attempts to classify intrusion"""

__author__  = "Dave Cameron, Raymond Xu, Nate Lemke"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# import sklearn as sk

# Import data
dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -1].values

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

# Train models


# Predict results


# Visualize training set


# Visualize test set