import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
from math import log
import time

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
dataFile = '8085_A1\\reduced_dataset_100000.json'

class NBClassifier():
    def __init__(self, alpha_s = 0.01, alpha_cuf = 0.01) -> None:
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
            counts.append(df.value_counts()[i+1])
        
        for i in range(5):
            self.count_dict[str(i+1)] = counts[i]
            self.priors_dict[str(i+1)] = log(counts[i]/total)
        print(self.priors_dict)
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
        st = time.time()
        print("training...")

        self.get_priors(y_train)
        X_processed = self.preprocessing(X_train)
        y_train = y_train.to_numpy()
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
        #Counting times each word appears in training data
        for i in range(len(X_processed)):
            label = y_train[i]
            tokens = X_processed[i].split()
            for token in tokens:
                self.words_dicts[str(label)][token] += 1

        print("training time: ", time.time() - st)

    def get_word_prob(self, word, label):
        numerator = self.words_dicts[label][word] + self.alpha_s
        denominator = self.count_dict[label] + (len(self.unique_words)*self.alpha_s)
        return log(numerator/denominator)

    def predict(self, X_test):
        results = []
        st = time.time()
        print("predicting...")

        X_processed = self.preprocessing(X_test)
        for text in X_processed:
            label_probs = []
            words = text.split()
            for label in range(1,6):
                label_probs.append(self.priors_dict[str(label)])
                for word in words:
                    if word in self.unique_words:
                        label_probs[label-1] += self.get_word_prob(word, str(label))
            results.append(label_probs.index(max(label_probs))+1)
        
        print("predicting time: ", time.time() - st)
        return results

        
    
def get_data(file):
    print("reading...")
    chunks = pd.read_json(file, lines=True, chunksize=1000000)
    df = pd.concat(chunks)
    df.drop(['review_id', 'user_id', 'business_id', 'date'], axis='columns', inplace=True)
    return df


if __name__ == "__main__":
    df = get_data(dataFile)
  
    X = df['text']
    y = df['stars']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/5, random_state=0)

    clf = NBClassifier(alpha_s=0.1)

    clf.train(X_train, y_train)
    pred = clf.predict(X_test)
    print(accuracy_score(y_true=y_test, y_pred=pred))
    print(confusion_matrix(y_true=y_test, y_pred=pred))

    """
    X = df['text']
    y = df['stars']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/5)

    X_train = preprocessing(X_train)
    cv = CountVectorizer()
    X_train = cv.fit_transform(X_train)
    X_test = preprocessing(X_test)
    X_test = cv.transform(X_test)

    def preprocessing(X):
    sw = set(stopwords.words('english'))
    lm = WordNetLemmatizer()
    print("preprocessing...")

    for index, text in enumerate(X):

        text = text.lower()
        tl = str.maketrans('','',string.punctuation)
        text = text.translate(tl)
        words = word_tokenize(text)

        l_words = []
        for word in words:
            if word not in sw:
                l_words.append(lm.lemmatize(word))

        text = ' '.join(l_words)
        
    return X

    def get_star_priors(df):
    total = len(df)
    num1 = df['stars'].value_counts()[1]
    num2 = df['stars'].value_counts()[2]
    num3 = df['stars'].value_counts()[3]
    num4 = df['stars'].value_counts()[4]
    num5 = df['stars'].value_counts()[5]
    return({1:num1,2:num2,3:num3,4:num4,5:num5},
           {1:num1/total,2:num2/total,3:num3/total,4:num4/total,5:num5/total})
    """
