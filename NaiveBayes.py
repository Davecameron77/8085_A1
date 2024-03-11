import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from collections import Counter
from math import log
import time

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
dataFile = '8085_A1/reduced_dataset_10000.json'

class NBClassifier():
    def __init__(self, alpha_s = 0.01, alpha_cuf = 0.00001) -> None:
        self.words_dicts = {}
        self.priors_dict = {}
        self.count_dict = {}
        self.unique_words = []
        self.alpha_s = alpha_s
        self.alpha_cuf = alpha_cuf
    
    """
    calculates the count and prior probability of each class.
    stores these in count_dict and priors_dict.
    """
    def get_priors(self, df):
        total = len(df)
        counts = []

        
        for i in range(5):
            counts.append(df.stars.value_counts()[i+1])
        
        for i in range(5):
            self.count_dict[str(i+1)] = counts[i]
            self.priors_dict[str(i+1)] = log(counts[i]/total)

        #getting counts of cool values in ranges
        self.count_dict['c0'] = (df.cool == 0).sum()
        self.count_dict['c1'] = (df.cool == 1).sum()
        self.count_dict['c2'] = ((df.cool >= 2) & (df.cool < 5)).sum()
        self.count_dict['c5'] = (df.cool >= 5).sum()
        #getting priors of cool values in ranges
        self.priors_dict['c0'] = log(self.count_dict['c0']/total)
        self.priors_dict['c1'] = log(self.count_dict['c1']/total)
        self.priors_dict['c2'] = log(self.count_dict['c2']/total)
        self.priors_dict['c5'] = log(self.count_dict['c5']/total)

        #getting counts of useful values in ranges
        self.count_dict['u0'] = (df.useful == 0).sum()
        self.count_dict['u1'] = (df.useful == 1).sum()
        self.count_dict['u2'] = ((df.useful >= 2) & (df.useful < 5)).sum()
        self.count_dict['u5'] = (df.useful >= 5).sum()
        #getting priors of useful values in ranges
        self.priors_dict['u0'] = log(self.count_dict['u0']/total)
        self.priors_dict['u1'] = log(self.count_dict['u1']/total)
        self.priors_dict['u2'] = log(self.count_dict['u2']/total)
        self.priors_dict['u5'] = log(self.count_dict['u5']/total)

        #getting counts of funny values in ranges
        self.count_dict['f0'] = (df.funny == 0).sum()
        self.count_dict['f1'] = (df.funny == 1).sum()
        self.count_dict['f2'] = ((df.funny >= 2) & (df.funny < 5)).sum()
        self.count_dict['f5'] = (df.funny >= 5).sum()
        #getting priors of funny values in ranges
        self.priors_dict['f0'] = log(self.count_dict['f0']/total)
        self.priors_dict['f1'] = log(self.count_dict['f1']/total)
        self.priors_dict['f2'] = log(self.count_dict['f2']/total)
        self.priors_dict['f5'] = log(self.count_dict['f5']/total)


    """
    processes text for use in classification.
    removes uppercase punctuation, and stopwords, lemmatizes result.
    """
    def preprocessing(self, X):
        sw = set(stopwords.words('english'))
        lm = WordNetLemmatizer()
        processed = []
        
        for text in X:

            text = text.lower()
            tl = str.maketrans('','',string.punctuation)
            text = text.translate(tl)
            tl = str.maketrans('','', string.digits)
            text = text.translate(tl)
            words = word_tokenize(text)

            l_words = []
            for word in words:
                if word not in sw:
                    l_words.append(lm.lemmatize(word))

            text = ' '.join(l_words)
            processed.append(text)

        return processed
    
    def train(self, X_train, y_train):
        cuf_keys = ['c0','c1','c2','c5','u0','u1','u2','u5','f0','f1','f2','f5']
        st = time.time()
        print("training...")

        self.get_priors(y_train)
        X_processed = self.preprocessing(X_train)
        y_stars = y_train.stars.values
        y_cool = y_train.cool.values
        y_useful = y_train.useful.values
        y_funny = y_train.funny.values

        #creating list of all unique words in training data
        for text in X_processed:
            words = text.split()
            for word in words:
                if word not in self.unique_words:
                    self.unique_words.append(word)

        #initializing dicts that contain word counts
        for i in range(5):
            self.words_dicts[str(i+1)] = {}
            for j in self.unique_words:
                self.words_dicts[str(i+1)][j] = 0

        for k in cuf_keys:
            self.words_dicts[k] = {}
            for j in self.unique_words:
                self.words_dicts[k][j] = 0

        #Counting times each word appears in training data
        for i in range(len(X_processed)):
            s_label = y_stars[i]

            if(y_cool[i] == 0): c_label = 'c0'
            elif(y_cool[i] == 1): c_label = 'c1'
            elif(2 >= y_cool[i] < 5): c_label = 'c2'
            elif(y_cool[i] > 5): c_label = 'c5'

            if(y_useful[i] == 0): u_label = 'u0'
            elif(y_useful[i] == 1): u_label = 'u1'
            elif(2 >= y_useful[i] < 5): u_label = 'u2'
            elif(y_useful[i] > 5): u_label = 'u5'

            if(y_funny[i] == 0): f_label = 'f0'
            elif(y_funny[i] == 1): f_label = 'f1'
            elif(2 >= y_funny[i] < 5): f_label = 'f2'
            elif(y_funny[i] > 5): f_label = 'f5'

            tokens = X_processed[i].split()
            for token in tokens:
                self.words_dicts[str(s_label)][token] += 1
                self.words_dicts[c_label][token] += 1
                self.words_dicts[u_label][token] += 1
                self.words_dicts[f_label][token] += 1

        print("training time: ", time.time() - st)

    def get_word_prob(self, word, label):
        if label in ['1','2','3','4','5']:
            numerator = self.words_dicts[label][word] + self.alpha_s
            denominator = self.count_dict[label] + (len(self.unique_words)*self.alpha_s)
        else:
            numerator = self.words_dicts[label][word] + self.alpha_cuf
            denominator = self.count_dict[label] + (len(self.unique_words)*self.alpha_cuf)
        return log(numerator/denominator)

    def predict(self, X_test):
        s_keys = ['1','2','3','4','5']
        c_keys = ['c0','c1','c2','c5','n']
        u_keys = ['u0','u1','u2','u5','n']
        f_keys = ['f0','f1','f2','f5','n']
        results = [[]for i in range(4)]
        st = time.time()
        print("predicting...")

        X_processed = self.preprocessing(X_test)
        for text in X_processed:
            s_probs,c_probs,u_probs,f_probs = ([]for i in range(4))
            words = text.split()

            for s_key,c_key,u_key,f_key in zip(s_keys,c_keys,u_keys,f_keys):
                s_cur_prob = self.priors_dict[s_key]
                if c_key != 'n':
                    c_cur_prob = self.priors_dict[c_key]
                    u_cur_prob = self.priors_dict[u_key]
                    f_cur_prob = self.priors_dict[f_key]
                for word in words:
                    if word in self.unique_words:
                        s_cur_prob += self.get_word_prob(word, s_key)
                        if c_key != 'n':
                            c_cur_prob += self.get_word_prob(word, c_key)
                            u_cur_prob += self.get_word_prob(word, u_key)
                            f_cur_prob += self.get_word_prob(word, f_key)
                s_probs.append(s_cur_prob)
                if c_key != 'n':
                    c_probs.append(c_cur_prob)
                    u_probs.append(u_cur_prob)
                    f_probs.append(f_cur_prob)
            results[0].append(s_probs.index(max(s_probs))+1)
            for i in range(1,4):
                match i:
                    case 1: max_index = c_probs.index(max(c_probs))
                    case 2: max_index = u_probs.index(max(u_probs))
                    case 3: max_index = f_probs.index(max(f_probs))
                match max_index:
                    case 0: pred = 0
                    case 1: pred = 1
                    case 2: pred = 2
                    case 3: pred = 5
                results[i].append(pred)

        print("predicting time: ", time.time() - st)

        return results

        
    
def get_data(file):
    print("reading...")
    chunks = pd.read_json(file, lines=True, chunksize=1000000)
    df = pd.concat(chunks)
    df.drop(['review_id', 'user_id', 'business_id', 'date'], axis='columns', inplace=True)
    return df

def evaluation(y_true, y_pred):
    star_counts = [0]*5
    c_total = 0
    u_total = 0
    f_total = 0

    total = len(y_true)

    s_true = y_true['stars'].values
    c_true = y_true['cool'].values
    u_true = y_true['useful'].values
    f_true = y_true['funny'].values

    s_pred = y_pred[0]
    c_pred = y_pred[1]
    u_pred = y_pred[2]
    f_pred = y_pred[3]

    for i in range(total):
        star_counts[abs(s_true[i] - s_pred[i])] += 1
        c_total += abs(c_pred[i] - c_true[i])
        u_total += abs(u_pred[i] - u_true[i])
        f_total += abs(f_pred[i] - f_true[i])
    

    print("Correct: {}\n1 off: {}\n2 off: {}\n3 off: {}\n4 off: {}".format((star_counts[0]/total)*100,(star_counts[1]/total)*100,
                              (star_counts[2]/total)*100,(star_counts[3]/total)*100,(star_counts[4]/total)*100))
    print("Cool off by an average of: {}".format(c_total/total))
    print("Useful off by an average of: {}".format(u_total/total))
    print("Funny off by an average of: {}".format(f_total/total))

if __name__ == "__main__":
    df = get_data(dataFile)

  
    X = df['text']
    y = df[['stars','cool','useful','funny']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/5, random_state=0)

    clf = NBClassifier(alpha_s=0.01, alpha_cuf=0.0001)

    clf.train(X_train, y_train)
    pred = clf.predict(X_test)
    evaluation(y_true=y_test, y_pred=pred)

