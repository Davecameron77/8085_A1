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
from sklearn.metrics import classification_report, precision_score
from sklearn import metrics
import pickle
from sklearn.preprocessing import MinMaxScaler
import argparse
from imblearn.over_sampling import SMOTE

PRINT_TRAINING_SCORE = True

class Classification_target(Enum):
    Label = 1
    Attack_cat = 2

class Classifier(Enum):
    RandomForestClassifier = 1
    LogisticRegression = 2
    KNearestNeighbors = 3

feature_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 
                'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 
                'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 
                'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 
                'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 
                'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
                'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 
                'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 
                'ct_dst_sport_ltm', 'ct_dst_src_ltm']
rfc_correlated_features = ['sttl', 'ct_state_ttl', 'dttl', 'ackdat', 'state', 
                           'tcprtt', 'proto']
rfc_important_features = ['dsport', 'ct_srv_dst', 'dbytes', 'sbytes', 'dmeansz', 
                          'smeansz', 'sport', 'Dpkts']
rfc_categorical_features = ['srcip', 'dstip']

Feat15 =          ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 
                   'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 
                   'ct_state_ttl', 'ct_srv_dst']
label = ['None', 'Generic', 'Fuzzers', 'Exploits', 'Dos', 'Reconnaissance', 'Analysis', 'Shellcode', 'Backdoors', 'Worms']

#region Dave Special

# @dave special
def analyze_feature_correlation():
    return
# @dave special
def hyperparameter_tuning():
    return

#endregion

#region Raymond 
def apply_knn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(algorithm='brute')
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    return y_test, y_predict

def apply_d_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_test, y_pred

def apply_logistic_regression(X_train, y_train, X_test, y_test):
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    return y_test, y_pred
    
def draw_heatmap(pca):
    df_comp = pd.DataFrame(pca.components_, columns=feature_cols)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_comp, cmap="Blues")
    plt.show()

def draw_diagram(pca):
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))
    pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'], columns=['PC'])
    explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
    cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])
    cumulative_variance = np.cumsum(np.round(pca.explained_variance_, decimals=3))
    df_explained_variance = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_explained_variance['PC'],
            y=df_explained_variance['Cumulative Variance'],
            marker=dict(size=15, color="LightSeaGreen"),
        ))
    fig.add_trace(
        go.Bar(
            x=df_explained_variance['PC'],
            y=df_explained_variance['Explained Variance'],
            marker=dict(color="RoyalBlue")
        ))


    fig.show()
#endregion

def create_model(filename="UNSW-NB15-BALANCED-TRAIN.csv"):
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
    df['is_ftp_login'] = df['is_ftp_login'].astype(bool)
    df['is_sm_ips_ports'] = df['is_sm_ips_ports'].astype(bool)
    df['ct_ftp_cmd'], _ = pd.factorize(df['ct_ftp_cmd'])
    df['attack_cat'] = df['attack_cat'].astype('str')
    df['attack_cat'] = df['attack_cat'].str.strip()
    df['attack_cat'] = df['attack_cat'].replace('Backdoor', 'Backdoors')
    df['attack_cat'] = df['attack_cat'].replace('nan', 'Benign')
    df['Label'] = df['Label'].astype(bool)
    codes, _ = pd.factorize(df['attack_cat'])
    df['attack_cat'] = codes

    return df

def apply_PCA(train,test):
    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(train)
    train = pca.transform(train) 
    test = pca.transform(test)
    return train, test

def df_preprocessing(df, classifier, target, apply_dimension_reduction, for_validation=False):
    scaler = None
    if classifier == Classifier.LogisticRegression:
        scaler = MinMaxScaler()
        x = scaler.fit_transform(df[Feat15])
    elif classifier == Classifier.KNearestNeighbors:
        x = df[feature_cols]
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    elif classifier == Classifier.RandomForestClassifier:
        x = df[rfc_correlated_features]
        x = pd.concat([x, df[rfc_important_features]], axis=1)
        x = pd.concat([x, df[rfc_categorical_features]], axis=1)
    else:
        if apply_dimension_reduction:
            x = df[Feat15]
        else: 
            x = df[feature_cols]
    
    if target == Classification_target.Label: 
        y = df.Label # Target variable
    elif target == Classification_target.Attack_cat:
        y = df.attack_cat
    if for_validation:
        return x, y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5, random_state=0)
    if classifier == Classifier.KNearestNeighbors and apply_dimension_reduction:
        x_train, x_test = apply_PCA(x_train, x_test)
    
    return x_train, x_test, y_train, y_test

def df_postprocessing(x_train, y_train):
    sm = SMOTE(random_state = 1, k_neighbors = 5)
    x_train, y_train = sm.fit_resample(x_train, y_train)
    return np.unique(y_train, return_counts=True)

def classify(x_train, x_test, y_train, classifier, model_loaded):
    if not model_loaded:
        classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    return y_predict

def validation(filename, classifier_enum, classifier, target, apply_dimension_reduction):
    df = create_model(filename)
    x, y = df_preprocessing(df, classifier_enum, target, apply_dimension_reduction, for_validation=True)
    if classifier_enum == Classifier.KNearestNeighbors:
        pca = PCA(n_components=10, svd_solver='full')
        pca.fit(x)
        x = pca.transform(x) 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0) 
    return classifier.predict(x_test), y_test
    
def print_result(y_test, y_predict, classification_target):
    if PRINT_TRAINING_SCORE:
        if classification_target == Classification_target.Label:
            print(classification_report(y_test, y_predict))
        else:
            print(classification_report(y_test, y_predict, target_names=label))

def main():
    feature_reduction = True 
    data_balance = True
    model_loaded = False
    optional_load_model_name = "" 
    parser = argparse.ArgumentParser()
    parser.add_argument('heldout_filename')  
    parser.add_argument('classification_method')   
    parser.add_argument('task')    
    args, unknown = parser.parse_known_args()
    heldout_filename = args.heldout_filename
    filename = ""
    classification_method = args.classification_method
    task = args.task

    classifier = None
    classifier_enum = None
    
    # Set classifier
    if classification_method == "RandomForestClassifier":
        classifier = RandomForestClassifier(n_estimators=1000, 
                                            criterion='entropy', max_depth=24, 
                                            min_samples_split=10,
                                            min_samples_leaf=2, 
                                            max_features=None, bootstrap=True, 
                                            n_jobs=-1)
        classifier_enum = Classifier.RandomForestClassifier 
    elif classification_method == "LogisticRegression":
        classifier = LogisticRegression(solver='saga', penalty='l1', C=5.0, max_iter=10000)
        classifier_enum = Classifier.LogisticRegression 
        model_loaded = True 
    elif classification_method == "KNearestNeighbors":
        classifier = KNeighborsClassifier() 
        classifier_enum = Classifier.KNearestNeighbors 

    # Set target
    classification_target = None 
    if task.lower() == 'label':
        classification_target = Classification_target.Label
    else:
        classification_target = Classification_target.Attack_cat
    
    # Load model in case of Logistic Regression
    if classifier_enum == Classifier.LogisticRegression :
         optional_load_model_name = unknown[0]
         classifier = pickle.load(open(optional_load_model_name, 'rb'))
    else:
        filename = unknown[0] 

    # Execute
    start_time = time.time()
    df = create_model(filename)
    x_train, x_test, y_train, y_test = df_preprocessing(df, classifier_enum, classification_target, feature_reduction)
    if data_balance:
        df_postprocessing(x_train, y_train)
    
    #training
    # print(x_test.shape)
    y_predict = classify(x_train, x_test, y_train, classifier, model_loaded)
    # print(classifier.classes_)
    print_result(y_test, y_predict, classification_target)
    execution_time = time.time() - start_time
    if PRINT_TRAINING_SCORE:
        print(f'Training completed in {execution_time} seconds')
    
    #validate
    y_predict, y_test = validation(heldout_filename,classifier_enum, classifier, classification_target, feature_reduction)
    print_result(y_test, y_predict, classification_target)
    validation_time = time.time() - start_time - execution_time
    print(f'Validation completed in {validation_time} seconds')

        
if __name__ == "__main__":
    main()
