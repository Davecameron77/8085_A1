#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from itertools import count

"""NIDS_DC.py: Analyzes collected network traffic and attempts to classify intrusion"""
__author__ = "Dave Cameron, Raymond Xu, Nate Lemke"

# Import data
dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', header=0, low_memory=False)
print("Dataset loaded")

# Transform object columns
# sport / dsport are really categorical, not numeric
# Booleans are treated as categorical
dataset['srcip'], _ = pd.factorize(dataset['srcip'])
dataset['sport'], _ = pd.factorize(dataset['sport'])
dataset['dstip'], _ = pd.factorize(dataset['dstip'])
dataset['dsport'], _ = pd.factorize(dataset['dsport'])
dataset['proto'], _ = pd.factorize(dataset['proto'])
dataset['state'], _ = pd.factorize(dataset['state'])
dataset['service'], _ = pd.factorize(dataset['service'])
dataset['ct_flw_http_mthd'], _ = pd.factorize(dataset['ct_flw_http_mthd'])
dataset['is_ftp_login'], _ = pd.factorize(dataset['is_ftp_login'])
dataset['ct_ftp_cmd'], _ = pd.factorize(dataset['ct_ftp_cmd'])
dataset['attack_cat'], _ = pd.factorize(dataset['attack_cat'])

# Dependant variable is both classification and prediction for now
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -2:-1]['attack_cat'].tolist()

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

# Train model
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)

# Select top N features
N = 10
features_to_use = {}
for i, feature in zip(count(), clf.feature_importances_):
    features_to_use[i] = feature
    # Clip to only max 5
    if len(features_to_use) == N+1:
        min_val = min(features_to_use.values())
        del features_to_use[list(features_to_use.keys())[list(features_to_use.values()).index(min_val)]]

# Retrain
indices = list(features_to_use.keys())
print(f'\nindices of {N} most important features: {indices}\n')
for index in indices:
    print(f'{dataset.columns[index]} : {dataset.dtypes.values[index]}')

X = dataset.iloc[:, np.r_[indices]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)
clf = clf.fit(X_train, y_train)

# Predict results
y_pred = clf.predict(X_test)
ac = accuracy_score(y_test, y_pred) * 100
print("\nwKNN-Classifier Binary Set-Accuracy is ", ac)

print("************************************************************")
print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
print(metrics.classification_report(y_test, y_pred))
print("************************************************************")

print("\tRunning covariance / correlation\t")
cov_dataset = np.array(X_train.iloc[0:10000, :])
cov_matrix = np.cov(X)
# cor_matrix = X.corr()
print(cov_matrix)
print()
# print(cor_matrix)