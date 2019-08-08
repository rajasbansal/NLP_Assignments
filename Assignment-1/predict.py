import pandas as pd
import numpy as np
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import string
import math
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import time
import pickle
import re
import sys

test_data = pd.read_json(sys.argv[2], lines=True)

x_test = test_data['review']
y_test = test_data['ratings']
print("Finished reading the test data")

analyzer = TfidfVectorizer(stop_words = "english").build_analyzer()
analyzer2 = TfidfVectorizer(stop_words = "english", ngram_range = (2,2)).build_analyzer()
tokenizer = TfidfVectorizer(stop_words = "english").build_tokenizer()
stop = stopwords.words('english')
stop.remove("not")

def negation(doc):
    l = tokenizer(doc)
    ret = []
    negl = 3
    nef = 0
    cnt = doc.count("!")
    if (cnt > 0):
        ret.extend(["ex_ma"]*cnt)
    if ("1 star" in doc):
        ret.append("1_star")
        ret.append("1_star")
        ret.append("1_star")
    if ("2 star" in doc or "2 stars" in doc):
        ret.append("2_star")
        ret.append("2_star")
        ret.append("2_star")
    if ("3 star" in doc or "3 stars" in doc):
        ret.append("3_star")
        ret.append("2_star")
        ret.append("2_star")
    if ("4 star" in doc or "4 stars" in doc):
        ret.append("4_star")
        ret.append("4_star")
        ret.append("4_star")
    if ("5 star" in doc or "5 stars" in doc):
        ret.append("5_star")
        ret.append("5_star")
        ret.append("5_star")        
    for i in range(len(l)):
#         if (not l[i].isupper()):
#             l[i] = l[i].lower()
        if (l[i] in stop or not(l[i].isalpha())):
            pass
        elif (l[i] == "not"):
            ret.append(l[i])
            nef += negl
        elif nef > 0:
            l[i] = "not_"+l[i]
            ret.append(l[i])
            nef -= 1
        else:
            ret.append(l[i])
    ret.extend(analyzer2(doc))
    return ret

f_i = open(sys.argv[1],"rb")
lr = pickle.load(f_i)
vocab = pickle.load(f_i)
f_i.close()

print ("Finished loading the pickle data")
x_test = vocab.transform(x_test)
print ("Finished vectorizing the data")

preddt = lr.predict(x_test)
pro_pre = lr.predict_proba(x_test)
treh= np.array([1,2,3,4,5]).reshape(5,1)
exp_class = np.matmul(pro_pre,treh)
print ("The mean squared error is :-")
# print(mean_squared_error(y_test,preddt)*len(y_test))
print(mean_squared_error(y_test,exp_class)*len(y_test))

f_o = open(sys.argv[3],"w")
for i in exp_class:
    f_o.write(str((i[0]))+"\n")
f_o.close()