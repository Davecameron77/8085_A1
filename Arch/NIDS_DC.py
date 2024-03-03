#!/usr/bin/env python3
from itertools import count

import numpy as np
import pandas as pd
import seaborn as sns
import sys
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn import metrics


# ************************* Generation ************************* #
def create_model(filename, n_estimators=1000, min_samples_split=10, min_samples_leaf=2, max_features=None, max_depth=24,
                 bootstrap=True):
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

    return RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                  max_features=max_features, bootstrap=True, n_jobs=6), df


# ************************* Analysis ************************* #
def perform_analysis(df, features_to_use=18, use_correlation=True):
    # Analysis
    print(f'Perfoming analysis using {features_to_use} features')
    analysis_set = df.copy()
    analysis_set['attack_cat'], _ = pd.factorize(analysis_set['attack_cat'])

    correlation = analysis_set.corr().values[-2:-1]
    covariance = analysis_set.cov().values[-2:-1]

    # Find the indices of the critical features
    correlation_keys = dict(enumerate(correlation[0]))
    # Remove label and attack_cat
    correlation_keys.popitem()
    correlation_keys.popitem()
    covariance_keys = dict(enumerate(covariance[0]))
    # Remove label and attack_cat
    covariance_keys.popitem()
    covariance_keys.popitem()

    correlation_keys = dict(sorted(correlation_keys.items(), key=lambda x: x[1]))
    covariance_keys = dict(sorted(covariance_keys.items(), key=lambda x: x[1]))

    if use_correlation:
        critical_keys = list(correlation_keys.keys())[features_to_use * -1:]
    else:
        critical_keys = list(covariance_keys.keys())[features_to_use * -1:]
    critical_keys = np.array(critical_keys)
    print('found critical keys: ', critical_keys)
    critical_keys = np.array([14, 29, 28, 26, 7, 9, 10, 4, 22, 36, 31, 5, 39, 2])
    X = df.iloc[:, np.r_[critical_keys]]
    X_Temp = analysis_set.iloc[:, np.r_[critical_keys, (len(df.columns) - 2)]]

    #TODO - Cleanup
    switched_on = False
    if switched_on:
        # Create PairPlot
        plot = sns.pairplot(X_Temp.head(10000), diag_kind='kde')
        plot.savefig("pairplot.png")
        print("Created PairPlot")

        # Create heatmap
        heatmap = sns.heatmap(X_Temp.head(10000), xticklabels=X.columns, yticklabels=X.columns)
        heatmap.get_figure().savefig("heatmap.png")
        print("Created HeatMap")

    return X


# ************************* Label Prediction ************************* #
def classify_label(df, clf, X, run_cross_validation=False):
    # Train label classifier
    print("Training label classifier")
    y = df.iloc[:, -1:]['Label'].tolist()

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, random_state=0)

    # Train model
    clf = clf.fit(X_train, y_train)

    # Predict results
    print("Predicting label")
    y_pred = clf.predict(X_test)
    print("************************************************************")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred))

    if run_cross_validation:
        print("************************************************************")
        print("Analyzing cross validation")
        cross_validation = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        print(f'Cross Validation Score:\n{cross_validation}')

    print("************************************************************\n")


# ************************* Attack_Cat Prediction ************************* #
def classify_attack_cat(df, clf, X, run_cross_validation=False):
    # Train attack_cat classifier
    print("Training attack_cat classifier")
    y = df.iloc[:, -2:-1]['attack_cat'].tolist()

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

    # Train model
    clf = clf.fit(X_train, y_train)

    # Predict results
    print("Predicting attack_cat")
    y_pred = clf.predict(X_test)
    print("************************************************************")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(metrics.classification_report(y_test, y_pred))

    if run_cross_validation:
        print("************************************************************")
        print("Analyzing cross validation")
        cross_validation = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        print(f'Cross Validation Score:\n{cross_validation}')

    print("************************************************************\n")


filename = ""
if len(sys.argv) > 0:
    filename = sys.argv[1]
else:
    exit(1)

start_time = time.time()

clf, df = create_model(filename, 1000, 10, 2, None, 24, True)
X = perform_analysis(df)
# classify_label(df, clf, X)
classify_attack_cat(df, clf, X)

execution_time = time.time() - start_time
print(f"Execution took {round(execution_time, 2)} seconds")
