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
import contractions

import re
import bz2
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC

from nltk.stem.porter import PorterStemmer

raw_train = bz2.BZ2File('C:/Users/caesa/Documents/projects/amzn_reviews/train.ft.txt.bz2')
raw_test = bz2.BZ2File('C:/Users/caesa/Documents/projects/amzn_reviews/test.ft.txt.bz2')

raw_train = [x.decode('utf-8') for x in raw_train]
raw_test = [x.decode('utf-8') for x in raw_test]

#normalize all words to lowercase
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in raw_train]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in raw_test]

#Display #word per review distribution
words_per_review = list(map(lambda x: len(x.split()), train_sentences))

sns.displot(words_per_review)
plt.xlabel('words')
plt.ylabel('Number of words')
plt.title('Word Frequency')
plt.show()

#display stats on words in corpus    
def list_stats(x):
    print('mean: {}, median {}, max words: {}, min words: {}'.
          format(round(np.mean(x),2),np.median(x),np.max(x), np.min(x)))


list_stats(words_per_review)
    # mean: 78.48, median 70.0, max words: 257, min words: 2

#Label each review
    #label_2 = good review, label 1 = bad
    #0 --> bad reviews
    #1 --> good reviews
    
# def review_type(x):
#     return 1 if re.match("__label__2", x) is not None else 0

train_label = [0 if x[:10] == '__label__1' else 1 for x in raw_train]
test_label = [0 if x[:10] == '__label__1' else 1 for x in raw_test]

#create data frame storing review updates
review_length_train = list(map(lambda x: len(x.split()), train_sentences))
review_length_test = list(map(lambda x: len(x.split()), test_sentences))

temp_train_data = list(zip(train_label, train_sentences, review_length_train))
temp_test_data = list(zip(test_label, test_sentences, review_length_test))

train_df = pd.DataFrame(temp_train_data, columns = ('good_review', 'review_raw','num_words'))
test_df = pd.DataFrame(temp_test_data, columns = ('good_review', 'review_raw','num_words'))

#cleanup irrelevant objects
del [words_per_review,train_label, test_label, review_length_train, review_length_test,
     temp_train_data, temp_test_data]

#compare words count by review type
sns.set_theme(style="darkgrid")
ax = sns.boxplot(x = 'good_review', y = 'num_words',
            hue="good_review",data = train_df)

#identify reviews with a URL
url_key_word = ['www.', '.gov', '.net', '.org', 'http','https']
def url_check(x):
    url_with_review = []
    
    for i in np.arange(len(x)):
        if any(key_word in x[i] for key_word in url_key_word):
            url_with_review.append(1)
        else:
            url_with_review.append(0)
            
    return url_with_review
        
#url instances
# np.sum(url_with_review)
#     #9422 instances

train_df['with_url'] = url_check(train_sentences)
test_df['with_url'] = url_check(test_sentences)

#check that texts are identified
# train_df.loc[train_df['with_url']==1]
#     #index 193, 436, 1797, 3599989 have url
# test_df.loc[test_df['with_url']==1]
#     #index 254, 1545, 2113, 398586 have url

#replace website with 'url'
url_key = "([^ ]+(?<=\.[a-z]{3}))"

def url_replace(x):
    output = []
    for i in np.arange(len(x)):
        if any(key_word in x[i] for key_word in url_key_word):
            x[i] = re.sub(url_key, 'url', x[i])
        output.append(x[i])
    return output


train_sentences = url_replace(train_sentences)
test_sentences = url_replace(test_sentences)

train_df['url_replace'] = train_sentences
test_df['url_replace'] = test_sentences


#replace digits
def digit_replace(x):
    for i in np.arange(len(x)):
        x[i] = re.sub(r"\d", '0', x[i])

train_sentences = digit_replace(train_sentences)
test_sentences = digit_replace(test_sentences)

train_df['digit_replace'] = digit_replace(train_sentences)
train_df['digit_replace'] = digit_replace(test_sentences)

#tokenize words
train_df['tokenized'] = train_df


#replace contractions
def contraction_replace(x):
     for i in np.arange(len(x)):
        x[i] = [contractions.fix(word) for word in x[i].split()]
    #test that contractions are extended            
    # aa = ["Hello World my name isn't caesar", "purple shoes couldn't be worn?!",
    #       "What's the point of it all?", "Tell Mark I said Hello!!!!"]
    # contraction_replace(aa)
    
train_df['contraction'] = contraction_replace(train_sentences)
test_df['contraction'] = contraction_replace(test_sentences)   


#identify and remove 'stopwords' (i.e. the, a)
stopwords = stopwords.words('english')
keep_words = ['no', 'but', 'not']
for words in keep_words:
    stopwords.remove(words)



#WIP


#stemming and lemmatize
#remove punctuations
#remove digits
#remove url







