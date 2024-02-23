#!/usr/bin/env python3
from itertools import count
from enum import Enum
import numpy as np
import pandas as pd
import sys
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
from sklearn.preprocessing import MinMaxScaler
# from imblearn.over_sampling import SMOTE

class Classification_target(Enum):
    Label = 1
    Attack_cat = 2

class Classifier(Enum):
    RandomForestClassifier = 1
    LogisticRegression = 2
    KNearestNeighbors = 3

feature_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state',
                    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss',
                    'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
                    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
                    'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime',
                    'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack',
                    'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
                    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
                    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
                    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

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
    df['attack_cat'] = df['attack_cat'].astype('str')
    df['attack_cat'] = df['attack_cat'].str.strip()
    codes, uniques = pd.factorize(df['attack_cat'])
    df = df.replace('Backdoor', 'Backdoors')
    
    df = df.apply(lambda x: pd.factorize(x)[0])

    return df, uniques 


def df_preprocessing(df, classifier, target):
    scaler = None
    if classifier == Classifier.LogisticRegression:
        scaler = MinMaxScaler()
        x = scaler.fit_transform(df[feature_cols])
    elif classifier == Classifier.KNearestNeighbors:
        x = df[feature_cols]
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        x = df[feature_cols]
    if target == Classification_target.Label: 
        y = df.Label # Target variable
    elif target == Classification_target.Attack_cat:
        y = df.attack_cat
    return train_test_split(x, y, test_size=1 / 5, random_state=0)

def classify(x_train, x_test, y_train, classifier):
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    return y_predict

def main(argv): 
    filename = ""
    classification_method = ""
    task = ""
    if len(sys.argv) > 0:
        filename = sys.argv[1]
        classification_method = sys.argv[2]
        task = sys.argv[3]
        # load_model_name = sys.argv[4]
    else:
        exit(1)

    classifier = None
    classifier_enum = None
    if classification_method == "RandomForestClassifier":
        classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=24, min_samples_split=10,
                                            min_samples_leaf=2, max_features=None, bootstrap=True, n_jobs=-1)
        classifier_enum = Classifier.RandomForestClassifier 
    elif classification_method == "LogisticRegression":
        classifier = pickle.load(open('Model_LR_lab', 'rb')) 
        classifier_enum = Classifier.LogisticRegression 
    elif classification_method == "KNearestNeighbors":
        classifier = KNeighborsClassifier() 
        classifier_enum = Classifier.KNearestNeighbors 

    classification_target = None 
    if task.lower() == 'label':
        classification_target = Classification_target.Label
    else:
        classification_target = Classification_target.Attack_cat
        classifier = pickle.load(open('Model_LR_atk', 'rb')) 

    start_time = time.time()
    df, uniques = create_model(filename)
    x_train, x_test, y_train, y_test = df_preprocessing(df, classifier_enum, classification_target)
    y_predict = classify(x_train, x_test, y_train, classifier)
    uniques = uniques.insert(0, ['None'])
    print(classification_report(y_test, y_predict))
    y_predict = execution_time = time.time() - start_time
    print(f'Execution took {round(execution_time, 2)} seconds')


if __name__ == "__main__":
    main(sys.argv)

