import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
import math
import time
import pickle
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors, Word2Vec
import os
import sys
def tokenize(i):
    zz = []
    for wrd in word_tokenize(i):
        if wrd not in string.punctuation:
            zz.append(wrd)
    return zz
print ("starting")
f = open(sys.argv[3],"rb")
new_model = pickle.load(f)
print ("done")
f.close()
start = time.time()
f = open(sys.argv[1], 'r')
train_data = []
train_answers = []

for i in f.readlines():
    temp = i.lower()
    int_list = temp.split('::::')
    train_answers.append(int_list[1])
    remove_target = int_list[0].split('<<target>>')
    tokenized_sentence = (tokenize(remove_target[0])) + ["--target_token--"] + (tokenize(remove_target[1]))
    train_data.append(tokenized_sentence)
f.close()

train_pool_of_answers = []
f = open(sys.argv[2], 'r')
for i in f.readlines():
    t = i.lower()
    t = (t.strip().split(" "))
    train_pool_of_answers.append(t)

window_size = 100
pos_window = 10
f_out = open("output.txt","w")
start = time.time()
not_found = 0
found = 0
lem_help = 0
for example_ind in range(len(train_data)):
    example = train_data[example_ind]
    ind = example.index("--target_token--")
    tmp = np.zeros(300)
    num = 0
    for word_ind in range(max(ind - window_size,0), ind + window_size + 1,1):
        if (word_ind < len(example)):
            if example[word_ind] in new_model.wv:
                tmp += new_model.wv[example[word_ind]]
                num += 1
                found += 1
            else:
                not_found +=1
    l = []
    if (num != 0):
        tmp /= num
    for i in range(len(train_pool_of_answers[example_ind])):
        if (train_pool_of_answers[example_ind][i] in new_model.wv):
            tr = tmp.dot(new_model.wv[train_pool_of_answers[example_ind][i]])
            l.append((tr,train_pool_of_answers[example_ind][i]))
        else: 
            l.append((0,train_pool_of_answers[example_ind][i]))
    l.sort(reverse = True)
    l = [b for (a,b) in l]
    for i in range(len(train_pool_of_answers[example_ind]) - 1):
        f_out.write(str(l.index(train_pool_of_answers[example_ind][i]) + 1) + " ")
        l[l.index(train_pool_of_answers[example_ind][i])] = -1
    f_out.write(str(l.index(train_pool_of_answers[example_ind][-1]) + 1) + "\n")
f_out.close() 