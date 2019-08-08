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

model = KeyedVectors.load_word2vec_format(sys.argv[3], binary=True)

def tokenize(i):
    zz = []
    for wrd in word_tokenize(i):
        if wrd not in string.punctuation:
            zz.append(wrd)
    return zz

start = time.time()
files_to_read = os.listdir(sys.argv[1])
data = ""
for i in range(len(files_to_read)):
    f = open(sys.argv[1] + files_to_read[i], 'r')
    data += f.read()
sentences = sent_tokenize(data.lower())
tokens = [tokenize(i) for i in sentences]
f.close()

start = time.time()
new_model = Word2Vec(tokens, size=300, min_count=1)
num_examples = new_model.corpus_count
new_model.build_vocab([list(model.vocab.keys())], update=True)
new_model.intersect_word2vec_format(sys.argv[3], binary=True, lockf=1.0)
new_model.train(tokens, total_examples=num_examples, epochs=new_model.epochs)
f = open(sys.argv[2],"wb")
pickle.dump(new_model,f)
f.close()