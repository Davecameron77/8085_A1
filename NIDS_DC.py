#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
import sys
import time
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import export_graphviz
from sklearn import metrics
from IPython import display


# TODO - In Progress
def advanced_analysis(X_train, X_test, y_train, y_test):
    # Number of trees in random forest
    n_estimators = np.linspace(100, 3000, int((3000 - 100) / 200) + 1, dtype=int)
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [1, 5, 10, 20, 50, 75, 100, 150, 200]
    # Minimum number of samples required to split a node
    min_samples_split = [1, 2, 5, 10, 15, 20, 30]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Criterion
    criterion = ['gini', 'entropy']
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion': criterion}

    rf_base = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf_base, param_distributions=random_grid, n_iter=30, cv=5, verbose=2,
                                   random_state=42, n_jobs=4)
    rf_random.fit(X_train, y_train)
    print(rf_random.score(X_train, y_train))
    print(rf_random.score(X_test, y_test))

    # Part Two
    # Create tree
    estimator = clf.estimators_[2]
    class_names = np.array(df['attack_cat'].unique())

    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, class_names=class_names, rounded=True, proportion=False, precision=2,
                    filled=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')
    display.Image(graph.create_png())


# ************************* Generation ************************* #
def create_model(filename, n_estimators=100, min_samples_leaf=1, max_depth=1000, max_features='sqrt',
                 max_leaf_nodes=None):
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
                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                  max_leaf_nodes=max_leaf_nodes, n_jobs=6), df


# ************************* Analysis ************************* #
def perform_analysis(df, features_to_use=15, use_correlation=True):
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
    X = df.iloc[:, np.r_[critical_keys]]
    X_Temp = analysis_set.iloc[:, np.r_[critical_keys, (len(df.columns) - 2)]]

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
        cross_validation = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=4)
        print(f'Cross Validation Score:\n{cross_validation}')

    print("************************************************************\n")


# ************************* Attack_Cat Prediction ************************* #
def classify_attack_cat(df, clf, X, run_cross_validation=False):
    # Train attack_cat classifier
    print("Training attack_cat classifier")
    y = df.iloc[:, -2:-1]['attack_cat'].tolist()

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, random_state=0)

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
        cross_validation = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=4)
        print(f'Cross Validation Score:\n{cross_validation}')

    print("************************************************************\n")


filename = ""
if len(sys.argv) > 0:
    filename = sys.argv[1]
else:
    exit(1)

start_time = time.time()

clf, df = create_model(filename, 100, 1, 30, 30, None)
X = perform_analysis(df)
classify_label(df, clf, X)
classify_attack_cat(df, clf, X)

execution_time = time.time() - start_time
print(f"Execution took {round(execution_time, 2)} seconds")
