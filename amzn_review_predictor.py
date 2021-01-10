#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:11:01 2021

@author: caesarphan
"""
#123
import pandas as pd
import numpy as np
import random

import re
import bz2
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

train_file_raw = bz2.BZ2File('C:/Users/caesa/Documents/projects/amzn_reviews/train.ft.txt.bz2')
test_file_raw = bz2.BZ2File('C:/Users/caesa/Documents/projects/amzn_reviews/test.ft.txt.bz2')

train_file = [x.decode('utf-8') for x in train_file_raw]
test_file = [x.decode('utf-8') for x in test_file_raw]

#for efficiency, randomly sample 300k to train, 10k to test
# random.seed(123)
# sample_train = random.sample(train_file, 300000)
# sample_test = random.sample(test_file, 10000)

#normalize all words - lowercase
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

#Display #word per review distribution
words_per_review = list(map(lambda x: len(x.split()), train_sentences))

sns.displot(words_per_review)
plt.xlabel('words')
plt.ylabel('Number of words')
plt.title('Word Frequency')
plt.show()

def list_stats(x):
    print('mean: {}, median {}, max words: {}, min words: {}'.
          format(round(np.mean(x),2),np.median(x),np.max(x), np.min(x)))

#display stats on words in corpus    
list_stats(words_per_review)
# mean: 78.48, median 70.0, max words: 257, min words: 2

#Label each review
    #label_2 = good review, label 1 = bad
    #0 --> bad reviews
    #1 --> good reviews
    
def review_type(x):
    return 1 if re.match("__label__2", x) is not None else 0

train_label = [0 if x[:10] == '__label__1' else 1 for x in train_file]
test_label = [0 if x[:10] == '__label__1' else 1 for x in test_file]
words_per_review = list(map(lambda x: len(x.split()), train_sentences))

temp_data = list(zip(words_per_review, train_label))
train_df = pd.DataFrame(temp_data, columns = ('num_words', 'good_review'))

#compare words count by review type
sns.set_theme(style="darkgrid")
ax = sns.boxplot(x = 'good_review', y = 'num_words',
            hue="good_review",data = train_df)

#remove url from reviews


#tokenize words
#stemming and lemmatize
#remove punctuations
#remove digits
#remove url







