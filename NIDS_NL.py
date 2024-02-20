import pandas as pd
import numpy as np
import time
import logging
import pickle

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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def FeatureSelection():
    dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)
    
    dataset['attack_cat'] = dataset['attack_cat'].replace('Backdoor', 'Backdoors')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers', 'Fuzzers')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers ', 'Fuzzers')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Reconnaissance ', 'Reconnaissance')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Shellcode ', 'Shellcode')

    o = (dataset.dtypes == 'object')
    object_cols = list(o[o].index)

    o = (dataset.dtypes != 'object')
    scale_cols = list(o[o].index)
    del scale_cols[-1]

    for label in object_cols:
        dataset[label], _ = pd.factorize(dataset[label])


    dataset[['ct_flw_http_mthd','is_ftp_login']] = dataset[['ct_flw_http_mthd','is_ftp_login']].fillna(0)
    
    mm = MinMaxScaler()

    dataset[scale_cols] = mm.fit_transform(dataset[scale_cols])

    print("Dataset read")

    X = dataset.iloc[:, :-2].values
    y = dataset.iloc[:, -2].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

    clf = RandomForestClassifier(max_depth=10)
    sel = RFE(clf, n_features_to_select=17 , step=10)

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

    dataset['attack_cat'] = dataset['attack_cat'].fillna('Benign')
    dataset[['ct_flw_http_mthd','is_ftp_login']] = dataset[['ct_flw_http_mthd','is_ftp_login']].fillna(0)

    dataset['attack_cat'] = dataset['attack_cat'].replace('Backdoor', 'Backdoors')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers', 'Fuzzers')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers ', 'Fuzzers')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Reconnaissance ', 'Reconnaissance')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Shellcode ', 'Shellcode')

    o = (dataset.dtypes == 'object')
    object_cols = list(o[o].index)

    le = LabelEncoder()
    for label in object_cols:
        #dataset[label] = le.fit_transform(dataset[label])
        dataset[label], _ = pd.factorize(dataset[label])


    

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

    atkFeat20 = ['sport', 'dsport', 'proto', 'state', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload',
'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'tcprtt', 'ackdat', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_src_dport_ltm']
    atkFeat15 = ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'ct_srv_dst']
    atkFeat10 = ['sport', 'dsport', 'sbytes', 'sttl', 'dttl', 'service', 'Dload', 'smeansz', 'dmeansz', 'ct_state_ttl']
    atkFeat5 = ['sport', 'dsport', 'sbytes', 'sttl', 'ct_state_ttl']

    tl= ['Benign', 'Generic', 'Fuzzers', 'Exploits', 'DOS', 'Reconnaissance', 'Backdoors', 'Analysis', 'Shellcode', 'Worms', ]

    dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)
    
    dataset['attack_cat'] = dataset['attack_cat'].replace('Backdoor', 'Backdoors')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers', 'Fuzzers')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers ', 'Fuzzers')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Reconnaissance ', 'Reconnaissance')
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Shellcode ', 'Shellcode')

    dataset['attack_cat'] = dataset['attack_cat'].fillna('Benign')
    dataset[['ct_flw_http_mthd','is_ftp_login']] = dataset[['ct_flw_http_mthd','is_ftp_login']].fillna(0)

    o = (dataset.dtypes == 'object')
    object_cols = list(o[o].index)


    print(dataset.head(10))

    print(dataset['attack_cat'].value_counts())

    for label in object_cols:
        dataset[label], _ = pd.factorize(dataset[label])

    
    cols = list(dataset.columns)
    cols.pop()
    cols.pop()

    mm = MinMaxScaler()
    dataset[cols] = mm.fit_transform(dataset[cols])

    print(dataset.head(10))

    X = dataset[atkFeat15].values
    y = dataset.iloc[:, -2].values

    print(dataset['attack_cat'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)
    
    
    t = time.process_time()

    clf = LogisticRegression()
  
    print("training...")
    clf.fit(X_train, y_train)

    print("predicting...")
    y_pred = clf.predict(X_test)

    print("////////////////////////////////")
    et = time.process_time() - t
    print("elapsed time: ", et)

    print("Accuracy for Atk is : {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred, target_names = tl))

    """
    print("starting grid...")
    params = [{'gamma': ['scale', 'auto'],
               'C' : [1.0, 0.1]}]
    
    grid = GridSearchCV(estimator=clf, param_grid=params, scoring='f1_macro', n_jobs=-1, verbose=3)

    grid.fit(X_train, y_train)

    print(grid.best_params_)

    pred = grid.predict(X_test)
    print(metrics.classification_report(y_test, pred))
    """
  
    #filename = 'SVCPoly.sav'
    #pickle.dump(clf, open(filename, 'wb'))

def loadClass():
    filename = 'SVCPoly.sav'
    clf = pickle.load(open(filename, 'rb'))

    atkFeat20 = ['sport', 'dsport', 'proto', 'state', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload',
'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'tcprtt', 'ackdat', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_src_dport_ltm']
    atkFeat15 = ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'ct_srv_dst']
    atkFeat10 = ['sport', 'dsport', 'sbytes', 'sttl', 'dttl', 'service', 'Dload', 'smeansz', 'dmeansz', 'ct_state_ttl']
    atkFeat5 = ['sport', 'dsport', 'sbytes', 'sttl', 'ct_state_ttl']

    dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)
    
    dataset['attack_cat'] = dataset['attack_cat'].replace('Backdoor', "Backdoors")
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers', "Fuzzers")
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Fuzzers ', "Fuzzers")
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Reconnaissance ', "Reconnaissance")
    dataset['attack_cat'] = dataset['attack_cat'].replace(' Shellcode ', "Shellcode")

    object_cols = list(o[o].index)

    for label in object_cols:
        dataset[label], _ = pd.factorize(dataset[label])


    dataset[['ct_flw_http_mthd','is_ftp_login']] = dataset[['ct_flw_http_mthd','is_ftp_login']].fillna(0)
    
    cols = list(dataset.columns)
    cols.pop()
    cols.pop()

    mm = MinMaxScaler()
    dataset[cols] = mm.fit_transform(dataset[cols])

    print("Dataset read")

    X = dataset[atkFeat15].values
    y = dataset.iloc[:, -2].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
    
    t = time.process_time()

    print("predicting...")
    y_pred = clf.predict(X_test)

    print("////////////////////////////////")
    et = time.process_time() - t
    print("elapsed time: ", et)

    print("Accuracy for Atk is : {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred))

def main():
    classifyAtk()

if __name__ == "__main__":
    main()