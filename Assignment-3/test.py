import time
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import sys
import pickle

f = open(sys.argv[1],"r")
lines = f.readlines()
tweet = []
examples = []
for i in range(len(lines)):
    if (not lines[i].strip()):
        examples.append(tweet)
        tweet = []
    else:
        z = lines[i].strip()
        tweet.append(z)

def length_feature(train_example, position):
    return len(train_example[position])

def full_word(train_example, position):
    return train_example[position].lower()
def last4(train_example, position):
    return train_example[position].lower()[-4:]
def last3(train_example, position):
    return train_example[position].lower()[-3:]
def last2(train_example, position):
    return train_example[position].lower()[-2:]
def last1(train_example, position):
    return train_example[position].lower()[-1:]    
def first4(train_example, position):
    return train_example[position].lower()[:4]
def first3(train_example, position):
    return train_example[position].lower()[:3]
def first2(train_example, position):
    return train_example[position].lower()[:2]
def first1(train_example, position):
    return train_example[position].lower()[:1]
def slash(train_example, position):
    return "/" in train_example[position].lower() and "http" not in train_example[position].lower()
def at(train_example, position):
    return "@" in train_example[position].lower() and "http" not in train_example[position].lower()
def back(train_example, position):
    return train_example[max(0,position-1)].lower()
def back4(train_example, position):
    return train_example[max(0,position-1)].lower()[-4:]
def back3(train_example, position):
    return train_example[max(0,position-1)].lower()[-3:]
def back2(train_example, position):
    return train_example[max(0,position-1)].lower()[-2:]
def forward(train_example, position):
    return train_example[min(position+1, len(train_example) - 1)].lower()
def forward4(train_example, position):
    return train_example[min(position+1, len(train_example) - 1)].lower()[-4:]
def forward3(train_example, position):
    return train_example[min(position+1, len(train_example) - 1)].lower()[-3:]
def forward2(train_example, position):
    return train_example[min(position+1, len(train_example) - 1)].lower()[-2:]

functions = [full_word, last4, last3, last2, first2, first3, first4, length_feature, back, forward]
c = 0
f_ex = []
for i in examples:
    tmp_ex = []
    for j in range(len(i)):
        temp_dict = {}
        for k in functions:
            temp_dict[k.__name__] = k(i,j)
        tmp_ex.append(temp_dict)
    f_ex.append(tmp_ex)

x_test = f_ex
f = open("model_best", "rb")
crf = pickle.load(f)
f.close()

f = open(sys.argv[2], "w")
y_pred = crf.predict(x_test)
for i in range(len(y_pred)):
    for j in range(len(y_pred[i])):
        f.write(examples[i][j] + " " + y_pred[i][j] + "\n")
    f.write("\n")
f.close()
