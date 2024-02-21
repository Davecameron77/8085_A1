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
df['attack_cat'], _ = pd.factorize(df['attack_cat'])
df = df.replace('Backdoor', 'Backdoors')
df['Label'] = df['Label'].astype(bool)

X = df.head(5000).iloc[:, np.r_[6, 7, 22, 24, 26, 2, 13, 14, 28, 29, 33, 4, 32, 5, 34, 10, 36, 9]]
y = df.head(5000).iloc[:, -2:-1]['attack_cat'].tolist()
feature_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
                 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
                 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
                 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
                 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
X_Train, y_train, X_Test, y_test = train_test_split(X, y, test_size=.5, random_state=0)

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
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print('\nRandom Grid\n******************')
print(random_grid)

# ************************* Box Three ************************* #
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3-fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_Train, y_train)

print('RF Best Params\n*****************************')
print(rf_random.best_params_)

