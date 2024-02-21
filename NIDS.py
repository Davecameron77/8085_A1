#!/usr/bin/env python3
from itertools import count

import numpy as np
import pandas as pd
import sys
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def create_model(filename):
    print(f'Loading data from {filename}')
    df = pd.read_csv(filename, header=0, low_memory=False, skipinitialspace=True)
    print("Dataset loaded\n")

    # Transform object columns
    # sport / dsport are really categorical, not numeric
    # Booleans are treated as categorical
    df['srcip'], _ = pd.factorize(df['srcip'])
    df['sport'], _ = pd.factorize(df['sport'])
    df['dstip'], _ = pd.factorize(df['dstip'])
    df['dsport'], _ = pd.factorize(df['dsport'])
    df['proto'], _ = pd.factorize(df['proto'])
    df['state'], _ = pd.factorize(df['state'])
    df['service'], _ = pd.factorize(df['service'])
    df['ct_flw_http_mthd'], _ = pd.factorize(df['ct_flw_http_mthd'])
    df['is_ftp_login'], _ = pd.factorize(df['is_ftp_login'])
    df['ct_ftp_cmd'], _ = pd.factorize(df['ct_ftp_cmd'])
    df['attack_cat'] = df['attack_cat'].fillna('')
    df['attack_cat'] = df['attack_cat'].astype('string')
    df['attack_cat'] = df['attack_cat'].str.strip()
    df = df.replace('Backdoor', 'Backdoors')
    df['Label'] = df['Label'].astype(bool)

    return df


def classify_label(dataframe, with_classifier=''):
    indices = [14, 29, 28, 26, 7, 9, 10, 4, 22, 36, 31, 5, 39, 2]

    x = dataframe.iloc[:, np.r_[indices]]
    y = df.iloc[:, -1:]['Label'].tolist()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5, random_state=0)
    if with_classifier == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=24, min_samples_split=10,
                                            min_samples_leaf=2, max_features=None, bootstrap=True, n_jobs=-1)
    if with_classifier == 'LogisticRegression':
        #TODO - Nate
        classifier = LogisticRegression(max_iter=1000)
    if with_classifier == 'KNearestNeighbors':
        #TODO - Raymond
        classifier = KNeighborsClassifier()

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    print('Accuracy: {:.2f}%\n'.format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred))


def classify_attack_cat(dataframe, with_classifier=''):
    indices = [14, 29, 28, 26, 7, 9, 10, 4, 22, 36, 31, 5, 39, 2]

    x = dataframe.iloc[:, np.r_[indices]]
    y = df.iloc[:, -1:]['attack_cat'].tolist()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 5, random_state=0)
    if with_classifier == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=24, min_samples_split=10,
                                            min_samples_leaf=2, max_features=None, bootstrap=True, n_jobs=-1)
    if with_classifier == 'LogisticRegression':
        #TODO - Nate
        classifier = LogisticRegression(max_iter=1000)
    if with_classifier == 'KNearestNeighbors':
        #TODO - Raymond
        classifier = KNeighborsClassifier()

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    print('Accuracy: {:.2f}%\n'.format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred))


filename = ""
if len(sys.argv) > 0:
    filename = sys.argv[1]
    classification_method = sys.argv[2]
    task = sys.argv[3]
    load_model_name = sys.argv[4]

else:
    exit(1)

start_time = time.time()

df = create_model(filename)
if task.lower() == 'label':
    classify_label(df, with_classifier=classification_method)
else:
    classify_attack_cat(df, with_classifier=classification_method)

execution_time = time.time() - start_time
print(f'Execution took {round(execution_time, 2)} seconds')