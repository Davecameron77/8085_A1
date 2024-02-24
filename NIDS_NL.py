import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import pickle


def preprocessing(datafile):

    print("Reading data...")
    dataset = pd.read_csv(datafile, low_memory=False)
    
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


def resampling(X,y):
    sampDict = {0:30000, 1:30000}
    us = RandomUnderSampler(sampling_strategy=sampDict)
    X_train, y_train = us.fit_resample(X, y)

    print("SMOTEin'...")
    sm = SMOTE(random_state = 1, k_neighbors = 5)
    X_train, y_train = sm.fit_resample(X, y)

    return X_train, y_train


def predict(X_train, y_train, X_test, y_test, show_con):

    tl= ['Benign', 'Generic', 'Fuzzers', 'Exploits', 'DOS', 'Recon', 'Backdoors', 'Analysis', 'Shellcode', 'Worms', ]

    t = time.process_time()

    clf = LogisticRegression(solver='saga', penalty='l1', n_jobs=-1, max_iter=100)

    print("training...")
    clf.fit(X_train, y_train)

    print("predicting...")
    y_pred = clf.predict(X_test)

    print("////////////////////////////////////////////////////////////////////////////////")
    et = time.process_time() - t
    print("elapsed time: ", et)

    print("Accuracy is : {:.2f}%\n".format(accuracy_score(y_test, y_pred) * 100))
    print("Classifier: Logistic Regression")
    print(classification_report(y_test, y_pred, target_names = tl))

    if show_con:
        plt.rcParams.update({'font.size': 8})
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tl)
        disp.plot()
        plt.show()


def FeatureSelection(datafile, show_con = False):

    dataset = preprocessing(datafile)

    X = dataset.iloc[:, :-2].values
    y = dataset.iloc[:, -2].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

    X_train, y_train = resampling(X_train, y_train)

    clf = RandomForestClassifier(max_depth=10)
    sel = RFE(clf, n_features_to_select=15 , step=10)

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


def classifyLab(datafile, show_con = False):

    Feat15 = ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'ct_srv_dst']

    dataset = preprocessing(datafile)

    X = dataset[Feat15].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)
    
    X_train, y_train = resampling(X_train, y_train)

    predict(X_train, y_train, X_test, y_test, show_con)


def classifyAtk(datafile, show_con = False):

    Feat15 = ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'ct_srv_dst']

    dataset = preprocessing(datafile)

    X = dataset[Feat15].values
    y = dataset.iloc[:, -2].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)
    
    X_train, y_train = resampling(X_train, y_train)

    predict(X_train, y_train, X_test, y_test, show_con)


def gridSearch(datafile):
    Feat15 = ['sport', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'service', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'ct_srv_dst']
    tl= ['Benign', 'Generic', 'Fuzzers', 'Exploits', 'DOS', 'Recon', 'Backdoors', 'Analysis', 'Shellcode', 'Worms', ]

    dataset = preprocessing(datafile)

    X = dataset[Feat15].values
    y = dataset.iloc[:, -2].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)
    
    X_train, y_train = resampling(X_train, y_train)

    clf = LogisticRegression()

    print("starting grid...")
    params = [{'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky' 'sag', 'saga' ],
               'C': [1.0, 3.0, 5.0],
               'max_iter': [100, 1000, 10000]}]
    
    grid = GridSearchCV(estimator=clf, param_grid=params, scoring='f1_macro', n_jobs=-1, verbose=3)

    grid.fit(X_train, y_train)

    print(grid.best_params_)

    pred = grid.predict(X_test)
    print(classification_report(y_test, pred, target_names=tl))

    params = [{'solver': ['saga' ],
               'penalty': ['l1', 'l2']}]
    
    grid = GridSearchCV(estimator=clf, param_grid=params, scoring='f1_macro', n_jobs=-1, verbose=3)

    grid.fit(X_train, y_train)

    print(grid.best_params_)

    pred = grid.predict(X_test)
    print(classification_report(y_test, pred, target_names=tl))
  

def main():
    datafile = 'UNSW-NB15-BALANCED-TRAIN.csv'
    classifyAtk(datafile)
    #gridSearch(datafile)


if __name__ == "__main__":
    main()