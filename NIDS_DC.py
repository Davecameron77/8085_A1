#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import sys

FEATURES_TO_USE = 10

filename = ""
if len(sys.argv) > 0:
    filename = sys.argv[1]
else:
    exit(1)

# Import data
print(f'Loading data from {filename}')
dataset = pd.read_csv(filename, header=0, low_memory=False)
print("Dataset loaded\n")

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
dataset['Label'], _ = pd.factorize(dataset['Label'])

# ************************* Analysis ************************* #

# Analysis
X = dataset.iloc[:, :-2].values
correlation = dataset.corr().values[-1]
covariance = dataset.cov().values[-1]

# Find the indices of the critical features
correlation_keys = {}
for i in range(0, len(correlation)):
    correlation_keys[i] = correlation[i]
covariance_keys = {}
for i in range(0, len(covariance)):
    covariance_keys[i] = covariance[i]

correlation_keys = dict(sorted(correlation_keys.items(), key=lambda x: x[1]))
covariance_keys = dict(sorted(covariance_keys.items(), key=lambda x: x[1]))

critical_keys = list(correlation_keys.keys())[FEATURES_TO_USE*-1:]

# ************************* Label Prediction ************************* #
# Train label classifier
print("Training label classifier")
y = dataset.iloc[:, -1:]['Label'].tolist()

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

# Train model
clf = RandomForestClassifier(n_estimators=10, max_depth=20, criterion='entropy')
clf = clf.fit(X_train, y_train)

# Predict results
y_pred = clf.predict(X_test)
print("************************************************************")
print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
print(metrics.classification_report(y_test, y_pred))
print("************************************************************\n")

# ************************* Attack_Cat Prediction ************************* #

# Train attack_cat classifier
print("Training attack_cat classifier")
y = dataset.iloc[:, -2:-1]['attack_cat'].tolist()

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

# Train model
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)

# Predict results
y_pred = clf.predict(X_test)
print("************************************************************")
print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
print(metrics.classification_report(y_test, y_pred))
print("************************************************************\n")