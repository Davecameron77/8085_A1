import pandas as pd
import numpy as np
import time
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn import metrics


def FeatureSelection():
    dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)
    dataset = dataset.replace('Backdoor', "Backdoors")

    o = (dataset.dtypes == 'object')
    object_cols = list(o[o].index)

    le = LabelEncoder()
    for label in object_cols:
        #dataset[label] = le.fit_transform(dataset[label])
        dataset[label], _ = pd.factorize(dataset[label])


    dataset[['ct_flw_http_mthd','is_ftp_login']] = dataset[['ct_flw_http_mthd','is_ftp_login']].fillna(0)
    

    print("Dataset read")

    X = dataset.iloc[:, :-2].values
    y = dataset.iloc[:, -2].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

    clf = RandomForestClassifier(max_depth=10)
    sel = RFE(clf, n_features_to_select=10 , step=10)

    t = time.process_time()
    sel = sel.fit(X_train, y_train)

    res = sel.get_support()
    indices = np.where(res)

    et = time.process_time() - t
    print("elapsed time: ", et)

    print("Selected features:")

    for i in indices[0]:
        print(dataset.columns[i])
    
    exit()

def classifyLab():
    labFeat10 = ['sbytes', 'dbytes', 'sttl', 'dttl', 'Dload', 'Dpkts', 'dmeansz', 'Dintpkt', 'synack', 'ct_state_ttl']
    labFeat5 = ['dbytes', 'sttl', 'Dload', 'dmeansz', 'ct_state_ttl']

    dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)
    dataset = dataset.replace('Backdoor', "Backdoors")
    o = (dataset.dtypes == 'object')
    object_cols = list(o[o].index)

    le = LabelEncoder()
    for label in object_cols:
        #dataset[label] = le.fit_transform(dataset[label])
        dataset[label], _ = pd.factorize(dataset[label])


    dataset[['ct_flw_http_mthd','is_ftp_login']] = dataset[['ct_flw_http_mthd','is_ftp_login']].fillna(0)
    

    print("Dataset read")

    X = dataset[labFeat10].values
    y = dataset.iloc[:, -1].values

    t = time.process_time()
 
    clf = LogisticRegression (max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
    clf = clf.fit(X_train, y_train)

    
    y_pred = clf.predict(X_test)
    ac = accuracy_score(y_test, y_pred) * 100

    print("////////////////////////////////")
    et = time.process_time() - t
    print("elapsed time: ", et)
    print("Logistic Regression Accuracy for Label is ", ac)
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred))
    print("////////////////////////////////")

def classifyAtk():

    atkFeat10 = ['sport', 'dsport', 'sbytes', 'sttl', 'dttl', 'service', 'Dload', 'smeansz', 'dmeansz', 'ct_state_ttl']
    atkFeat5 = ['sport', 'dsport', 'sbytes', 'sttl', 'ct_state_ttl']

    dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)
    
    dataset = dataset.replace('Backdoor', "Backdoors")
    o = (dataset.dtypes == 'object')
    object_cols = list(o[o].index)

    le = LabelEncoder()
    for label in object_cols:
        #dataset[label] = le.fit_transform(dataset[label])
        dataset[label], _ = pd.factorize(dataset[label])


    dataset[['ct_flw_http_mthd','is_ftp_login']] = dataset[['ct_flw_http_mthd','is_ftp_login']].fillna(0)
 

    print("Dataset read")

    X = dataset[atkFeat10].values
    y = dataset.iloc[:, -2].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
    
    t = time.process_time()

    clf = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='newton-cg', C=1)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("LR-Multi-i1000-c1-new////////////////////////////////")
    et = time.process_time() - t
    print("elapsed time: ", et)

    print("Accuracy for Atk is : {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred))



def main():
    classifyAtk()

if __name__ == "__main__":
    main()