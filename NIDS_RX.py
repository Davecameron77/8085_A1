#!/usr/local/bin/python3
import pandas as pd
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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
    df.head()
    df = df.apply(lambda x: pd.factorize(x)[0])
    return df


def train_model(df):
    

    X = df[feature_cols] # Features
    # X = apply_PCA(X)
    y = df.attack_cat # Target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: {:.2f}%\n".format(accuracy_score(y_test, y_pred) * 100))
    print(classification_report(y_test, y_pred))

def apply_PCA(df):
    scaler = StandardScaler()
    x = scaler.fit_transform(df)
    pca = PCA(n_components=8)
    principalDf = pca.fit_transform(x)
    return principalDf

def main():
    start_time = time.time()
    df = to_df(file="UNSW-NB15-BALANCED-TRAIN.csv")
    train_model(df)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()

