#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
X = dataset.iloc[:, 4:-2].values
y = dataset.iloc[:, -2:-1]['attack_cat'].tolist()

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

# Train model
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)
# Indicate measured importance
print(clf.feature_importances_)

# Predict results
y_pred = clf.predict(X_test)
ac = accuracy_score(y_test, y_pred) * 100
print("KNN-Classifier Binary Set-Accuracy is ", ac)
