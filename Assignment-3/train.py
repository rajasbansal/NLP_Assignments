import time
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import scipy
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import pickle
f = open("train.txt","r")
lines = f.readlines()
tweet = []
lab_tweet = []
examples = []
labels = []
for i in range(len(lines)):
    if (not lines[i].strip()):
        examples.append(tweet)
        labels.append(lab_tweet)
        tweet = []
        lab_tweet = []
    else:
        z = lines[i].strip()
        z = z.split(" ")
        tweet.append(z[0])
        lab_tweet.append(z[1])
start = time.time()
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

x_train = f_ex
y_train = labels
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

start = time.time()
crf = sklearn_crfsuite.CRF(
    algorithm='pa',
    all_possible_transitions=True
)
crf.fit(x_train, y_train)
f = open("model_best", "wb")
pickle.dump(crf,f)
f.close()