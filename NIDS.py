#!/usr/bin/env python3
from itertools import count
from enum import Enum
import numpy as np
import pandas as pd
import sys
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
from sklearn.preprocessing import MinMaxScaler
import argparse
from imblearn.over_sampling import SMOTE

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

indices = [14, 29, 28, 26, 7, 9, 10, 4, 22, 36, 31, 5, 39, 2]
reduced_feature = ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'ct_srv_dst']
Feat15 = ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'ct_srv_dst']

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
    df['attack_cat'] = df['attack_cat'].astype('str')
    df['attack_cat'] = df['attack_cat'].str.strip()
    codes, uniques = pd.factorize(df['attack_cat'])
    df['attack_cat'] = codes
    df = df.replace('Backdoor', 'Backdoors')

    return df, uniques 

def apply_PCA(train,test):
    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(train)
    train = pca.transform(train) 
    test = pca.transform(test)
    return train, test

def df_preprocessing(df, classifier, target, apply_dimension_reduction):
    scaler = None
    if classifier == Classifier.LogisticRegression:
        scaler = MinMaxScaler()
        x = scaler.fit_transform(df[Feat15])
    elif classifier == Classifier.KNearestNeighbors:
        x = df[feature_cols]
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        if apply_dimension_reduction:
            x = df[reduced_feature]
        else: 
            x = df[feature_cols]
    
    if target == Classification_target.Label: 
        y = df.Label # Target variable
    elif target == Classification_target.Attack_cat:
        y = df.attack_cat
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 5, random_state=0)

    if classifier == Classifier.KNearestNeighbors and apply_dimension_reduction:
        x_train, x_test = apply_PCA(x_train, x_test)
    
    return x_train, x_test, y_train, y_test

def df_postprocessing(x_train, y_train):
    sm = SMOTE(random_state = 1, k_neighbors = 5)
    x_train, y_train = sm.fit_resample(x_train, y_train)
    unique, counts = np.unique(y_train, return_counts=True)

def classify(x_train, x_test, y_train, classifier, model_loaded):
    if not model_loaded:
        classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    return y_predict

def main(argv):
    feature_reduction = True
    data_balance = True
    model_loaded = False
    optional_load_model_name = "" 
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')  
    parser.add_argument('classification_method')   
    parser.add_argument('task')    
    args, unknown = parser.parse_known_args()
    filename = args.filename
    classification_method = args.classification_method
    task = args.task

    classifier = None
    classifier_enum = None
    if len(unknown) == 1:
         optional_load_model_name = unknown[0]
         classifier = pickle.load(open(optional_load_model_name, 'rb'))
         model_loaded = True 

    if classification_method == "RandomForestClassifier":
        classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=24, min_samples_split=10,
                                            min_samples_leaf=2, max_features=None, bootstrap=True, n_jobs=-1)
        classifier_enum = Classifier.RandomForestClassifier 
    elif classification_method == "LogisticRegression":
        LogisticRegression(solver='saga', penalty='l1', C=5.0, max_iter=10000)
        classifier_enum = Classifier.LogisticRegression 
    elif classification_method == "KNearestNeighbors":
        classifier = KNeighborsClassifier() 
        classifier_enum = Classifier.KNearestNeighbors 

    classification_target = None 
    if task.lower() == 'label':
        classification_target = Classification_target.Label
    else:
        classification_target = Classification_target.Attack_cat

    start_time = time.time()
    df, uniques = create_model(filename)
    x_train, x_test, y_train, y_test = df_preprocessing(df, classifier_enum, classification_target, feature_reduction)
    if data_balance:
        df_postprocessing(x_train, y_train)
    y_predict = classify(x_train, x_test, y_train, classifier, model_loaded)
    if classification_target == Classification_target.Label:
        print(classification_report(y_test, y_predict))
    else:
        print(classification_report(y_test, y_predict, target_names=uniques))
    y_predict = execution_time = time.time() - start_time
    print(f'Execution took {round(execution_time, 2)} seconds')


if __name__ == "__main__":
    main(sys.argv)

