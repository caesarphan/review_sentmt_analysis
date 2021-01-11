#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:11:01 2021

@author: caesarphan
"""

import pandas as pd
import numpy as np
import random
import contractions
import string

import re
import bz2
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') 
from nltk.corpus import stopwords, wordnet
# nltk.download('stopwords')
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC

# from nltk.stem.porter import PorterStemmer

data_train = bz2.BZ2File('C:/Users/caesa/Documents/projects/amzn_reviews/train.ft.txt.bz2')
data_test = bz2.BZ2File('C:/Users/caesa/Documents/projects/amzn_reviews/test.ft.txt.bz2')

data_train = [x.decode('utf-8') for x in data_train]
data_test = [x.decode('utf-8') for x in data_test]

#for efficiency, randomly sample 300k to train, 10k to test
random.seed(123)
size_train = (len(data_train)/40)
size_test = (len(data_test)/40)

sample_train = random.sample(data_train, int(size_train))
sample_test = random.sample(data_test, int(size_test))

#normalize all words to lowercase
train_sent = [x.split(' ', 1)[1][:-1].lower() for x in sample_train]
test_sent = [x.split(' ', 1)[1][:-1].lower() for x in sample_test]

#Display #word per review distribution
words_per_review = list(map(lambda x: len(x.split()), train_sent))

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

train_label = [0 if x[:10] == '__label__1' else 1 for x in sample_train]
test_label = [0 if x[:10] == '__label__1' else 1 for x in sample_test]

#create data frame storing review updates
review_length_train = list(map(lambda x: len(x.split()), train_sent))
review_length_test = list(map(lambda x: len(x.split()), test_sent))

temp_train_data = list(zip(train_label, train_sent, review_length_train))
temp_test_data = list(zip(test_label, test_sent, review_length_test))

train_df = pd.DataFrame(temp_train_data, columns = ('good_review', 'review_raw','num_words'))
test_df = pd.DataFrame(temp_test_data, columns = ('good_review', 'review_raw','num_words'))

#cleanup irrelevant objects
del [size_train, size_test, words_per_review,train_label, test_label,
     review_length_train, review_length_test,
     temp_train_data, temp_test_data, sample_train, sample_test]

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

train_df['with_url'] = url_check(train_sent)
test_df['with_url'] = url_check(test_sent)

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


train_sent = url_replace(train_sent)
test_sent = url_replace(test_sent)

train_df['url_replace'] = train_sent
test_df['url_replace'] = test_sent

del [url_key,url_key_word]

#replace digits
def digit_replace(corpus):
    
    output = []
    
    for docs in corpus:       
        
        temp_list = []
        for i,word in enumerate(docs.split(' ')):
            temp_list.extend([word if word.isdigit() == False else '0'])
            
        output.append(' '.join(temp_list))                   
    return output

#Test function works
    # aa = test_sent.copy()
    # aa = digit_replace(aa)
    # data_test[39993]
    # aa[39993]

train_sent = digit_replace(train_sent)
test_sent = digit_replace(test_sent)

train_df['digit_replace'] = digit_replace(train_sent)
test_df['digit_replace'] = digit_replace(test_sent)

#replace contractions
def contrac_replace(x):
    output = []
    for i in np.arange(len(x)):
        x[i] = [contractions.fix(word) for word in x[i].split()]
        output.append(' '.join(x[i]))
    return output
   
    #test that contractions are extended            
    # aa = ["Hello World my name isn't caesar", "purple shoes couldn't be worn?!",
    #       "What's the point of it all?", "Tell Mark I said Hello!!!!"]
    # contraction_replace(aa)
    
    
    #test contractsions index 26
        # aa_test = test_sent.copy()
        # aa_test_v2 = contrac_replace(aa_test)

train_sent = contrac_replace(train_sent)
test_sent = contrac_replace(test_sent)
    
train_df['not_contrac'] = train_sent
test_df['not_contrac'] = test_sent

#tokenize words

train_sent = [word_tokenize(doc) for doc in train_sent]
test_sent = [word_tokenize(doc) for doc in test_sent]

train_df['tokenized'] = train_sent
test_df['tokenized'] = test_sent


#WIP
#remove punctuations
punc = string.punctuation

def remove_punc(corpus):
    output = []
    
    for docs in corpus:       
        temp_list = [word for word in docs if word not in punc]
            
        # temp_list = []
        # for word in enumerate(docs):
        #     temp_list.extend(word) if word not in punc            
            
        output.append(temp_list)                   
    return output
#Test
    # aa_trial = test_sent.copy()
    # aa_test = remove_punc(aa_trial)
    # ' '.join(aa_test[60])
    # ' '.join(aa_trial[60])

train_sent = remove_punc(train_sent)
test_sent = remove_punc(test_sent)

train_df['no_punc'] = train_sent
test_df['no_punc'] = test_sent

del [punc, punctuation]

#identify and remove 'stopwords' (i.e. the, a)
stopwords = stopwords.words('english')
keep_words = ['no', 'but', 'not']
stopwords = [word for word in stopwords if word not in keep_words]

train_df['rmv_stopwords'] = train_df['no_punc'].apply(lambda x: [word for word in x if word not in stopwords])
test_df['rmv_stopwords'] = test_df['no_punc'].apply(lambda x: [word for word in x if word not in stopwords])

del [stopwords, keep_words]


#Part of Speech
train_df['pos_tag'] = train_df['rmv_stopwords'].apply(nltk.pos_tag)
test_df['pos_tag'] = test_df['rmv_stopwords'].apply(nltk.pos_tag)

#wordnet pos tag
def pos_tagger(nltk_tag): 
    if nltk_tag.startswith('J'): 
        return wordnet.ADJ 
    elif nltk_tag.startswith('V'): 
        return wordnet.VERB 
    elif nltk_tag.startswith('N'): 
        return wordnet.NOUN 
    elif nltk_tag.startswith('R'): 
        return wordnet.ADV 
    else:           
        return None

train_df['wnl_pos'] = train_df['pos_tag'].apply(lambda x: [(word, pos_tagger(tag)) for (word, tag) in x])
test_df['wnl_pos'] = test_df['pos_tag'].apply(lambda x: [(word, pos_tagger(tag)) for (word, tag) in x])

#Lemmatize

# train_df['lemmatized'] = train_df['wnl_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
# test_df['lemmatized'] = test_df['wnl_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
wnl = WordNetLemmatizer()

def lemmatized(doc):
    
    lemmatized_sentence = [] 
    for word, tag in doc: 
        if tag is None: 
            # if there is no available tag, append the token as is 
            lemmatized_sentence.append(word) 
        else:         
            # else use the tag to lemmatize the token 
            lemmatized_sentence.append(wnl.lemmatize(word, tag)) 
    return lemmatized_sentence

train_df['lemmatized']= train_df['wnl_pos'].apply(lemmatized)
test_df['lemmatized']= test_df['wnl_pos'].apply(lemmatized)
