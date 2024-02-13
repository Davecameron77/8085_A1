#!/usr/local/bin/python3
import pandas as pd
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier

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

def to_df(file):
    df = pd.read_csv(file, header=0, low_memory=False)
    # df = df.apply(lambda x: pd.factorize(x)[0])
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
    codes, uniques = pd.factorize(df['attack_cat'])
    df['attack_cat'] = codes
    return df, codes, uniques 


def train_model(df):
    #scaler = StandardScaler()
    #df = scaler.fit_transform(df)
    X = df[feature_cols] # Features
    # y = df.Label # Target variable
    y = df.attack_cat
    # X = apply_PCA(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)
    return apply_knn(X_train, y_train, X_test, y_test)
    #return apply_d_tree(X_train, y_train, X_test, y_test)
    # return apply_logistic_regression(X_train, y_train, X_test, y_test)

def apply_knn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    return y_test, y_predict

def apply_d_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_test, y_pred

def apply_logistic_regression(X_train, y_train, X_test, y_test):
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    return y_test, y_pred

def apply_PCA(X):
    pca = PCA(n_components=2)
    principal_df = pca.fit_transform(X)
    return principal_df

def main():
    start_time = time.time()
    df, codes, uniques = to_df(file="UNSW-NB15-BALANCED-TRAIN.csv")
    y_test, y_pred = train_model(df)
    print(precision_score(y_test, y_pred, average = 'macro'))
    print(precision_score(y_test, y_pred, average = 'micro'))
    print(classification_report(y_test, y_pred))
    # print_attack_cat_classification_report_csv(report_str, uniques, codes)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()

