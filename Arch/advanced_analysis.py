#!/usr/bin/env python3
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV

start_time = time.time()

# region Default

df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', header=0, low_memory=False, skipinitialspace=True)
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
df['attack_cat'] = df['attack_cat'].str.strip()
df['attack_cat'] = df['attack_cat'].replace('Backdoor', 'Backdoors')
df['attack_cat'], _ = pd.factorize(df['attack_cat'])
df['Label'] = df['Label'].astype(bool)

features = ['dur', 'sbytes', 'smeansz', 'trans_depth', 'Sjit', 'dstip', 'service', 'Sload', 'Stime', 'Ltime', 'synack',
            'proto', 'tcprtt', 'state', 'ackdat', 'dttl', 'ct_state_ttl', 'sttl']

x = df[features].head(5000)
y = df.attack_cat.head(5000)
train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=.2, random_state=0)

# endregion

# ************************* Box One ************************* #
rf = RandomForestRegressor(random_state=42)
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
print(rf.get_params())

# ************************* Box Two ************************* #
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt', None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}
print('\nRandom Grid\n******************')
print(random_grid)

# ************************* Box Three ************************* #
# Use the random grid to search for best hyperparameters
# First create the base model to tune
# rf = RandomForestRegressor()
# Random search of parameters, using 3-fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the random search model
rf_random.fit(train_features, train_labels)

print('RF Best Params\n*****************************')
print(rf_random.best_params_)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mean_error = np.mean(errors)
    mean_label = np.mean(test_labels)
    mape = 100 * (mean_error / mean_label)
    accuracy = 100 - mape
    print('\nModel Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


base_model = RandomForestRegressor(n_estimators=10, random_state=42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

# ************************* Box Four ************************* #

param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [1, 2, 3],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(train_features, train_labels)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_labels)

print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))

print('\nGridSearch Best Params\n*****************************')
print(grid_search.best_params_)
