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

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def create_model(filename):
    """
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
    """
    ############################################
    print("Reading data...")
    dataset = pd.read_csv(filename, low_memory=False)
    
    dataset['attack_cat'] = dataset['attack_cat'].replace('Backdoor', 'Backdoors')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers', 'Fuzzers')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers ', 'Fuzzers')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Reconnaissance ', 'Reconnaissance')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Shellcode ', 'Shellcode')

    dataset['attack_cat'] = dataset['attack_cat'].fillna('Benign')
    dataset[['ct_flw_http_mthd','is_ftp_login']] = dataset[['ct_flw_http_mthd','is_ftp_login']].fillna(0)

    o = (dataset.dtypes == 'object')
    object_cols = list(o[o].index)

    for label in object_cols:
        dataset[label], _ = pd.factorize(dataset[label])

    cols = list(dataset.columns)
    cols.pop()
    cols.pop()

    mm = MinMaxScaler()
    dataset[cols] = mm.fit_transform(dataset[cols])

    return dataset
    ############################################

    return df


def classify_label(dataframe, with_classifier=''):
    indices = [14, 29, 28, 26, 7, 9, 10, 4, 22, 36, 31, 5, 39, 2]
    Feat15 = ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'ct_srv_dst']


    #x = dataframe.iloc[:, np.r_[indices]]
    #y = df.iloc[:, -1:]['Label'].tolist()

    ######################################
    x = dataframe[Feat15].values
    y = dataframe.iloc[:, -1].values
    ######################################

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5, random_state=0)

    ##################################################
    sampDict = {0:30000, 1:30000}
    us = RandomUnderSampler(sampling_strategy=sampDict)
    x_train, y_train = us.fit_resample(x_train, y_train)
    print("SMOTEin'...")
    sm = SMOTE(random_state = 1, k_neighbors = 5)
    x_train, y_train = sm.fit_resample(x_train, y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    print(np.asarray((unique,counts)).T)
    ################################################

    if with_classifier == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=24, min_samples_split=10,
                                            min_samples_leaf=2, max_features=None, bootstrap=True, n_jobs=-1)
    if with_classifier == 'LogisticRegression':
        classifier = LogisticRegression(max_iter=1000, solver='sag', n_jobs = -1)
    if with_classifier == 'KNearestNeighbors':
        #TODO - Raymond
        classifier = KNeighborsClassifier()

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    print('Accuracy: {:.2f}%\n'.format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred))


def classify_attack_cat(dataframe, with_classifier=''):
    indices = [14, 29, 28, 26, 7, 9, 10, 4, 22, 36, 31, 5, 39, 2]
    Feat15 = ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'ct_srv_dst']


    #x = dataframe.iloc[:, np.r_[indices]]
    #y = df.iloc[:, -2]['attack_cat'].tolist()

    ######################################
    x = dataframe[Feat15].values
    y = dataframe.iloc[:, -2].values
    ######################################
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 5, random_state=0)

    ##################################################
    sampDict = {0:30000, 1:30000}
    us = RandomUnderSampler(sampling_strategy=sampDict)
    x_train, y_train = us.fit_resample(x_train, y_train)
    print("SMOTEin'...")
    sm = SMOTE(random_state = 1, k_neighbors = 5)
    x_train, y_train = sm.fit_resample(x_train, y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    print(np.asarray((unique,counts)).T)
    ################################################

    if with_classifier == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=24, min_samples_split=10,
                                            min_samples_leaf=2, max_features=None, bootstrap=True, n_jobs=-1)
    if with_classifier == 'LogisticRegression':
        classifier = LogisticRegression(max_iter=1000, solver='sag', n_jobs = -1)
    if with_classifier == 'KNearestNeighbors':
        #TODO - Raymond
        classifier = KNeighborsClassifier()

    print("Training...")
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    print('Accuracy: {:.2f}%\n'.format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred))


filename = ""
"""
if len(sys.argv) > 0:
    filename = sys.argv[1]
    classification_method = sys.argv[2]
    task = sys.argv[3]
    load_model_name = sys.argv[4]

else:
    exit(1)
"""

#####################################################
filename = 'UNSW-NB15-BALANCED-TRAIN.csv'
classification_method = 'RandomForestClassifier'
task = 'atk_cat'
#####################################################

start_time = time.time()

df = create_model(filename)
if task.lower() == 'label':
    classify_label(df, with_classifier=classification_method)
else:
    classify_attack_cat(df, with_classifier=classification_method)

execution_time = time.time() - start_time
print(f'Execution took {round(execution_time, 2)} seconds')