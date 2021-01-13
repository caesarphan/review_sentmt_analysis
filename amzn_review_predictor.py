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
# nltk.download('wordnet') 
from nltk.corpus import stopwords, wordnet
# nltk.download('stopwords')
# nltk.download('punkt')
from string import punctuation
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn import metrics
# from sklearn.svm import SVC

# from nltk.stem.porter import PorterStemmer

#windows
data_train = bz2.BZ2File('C:/Users/caesa/Documents/projects/amzn_reviews/train.ft.txt.bz2')
data_test = bz2.BZ2File('C:/Users/caesa/Documents/projects/amzn_reviews/test.ft.txt.bz2')

# #macOS
# data_train = bz2.BZ2File('/Users/caesarphan/Documents/projects/amazon_reviews/train.ft.txt.bz2')
# data_test = bz2.BZ2File('/Users/caesarphan/Documents/projects/amazon_reviews/test.ft.txt.bz2')


data_train = [x.decode('utf-8') for x in data_train]
data_test = [x.decode('utf-8') for x in data_test]

#for efficiency, randomly sample 300k to train, 10k to test
random.seed(123)
# size_train = (len(data_train)/40)
# size_test = (len(data_test)/40)

size_train = (1000)
size_test = (100)

sample_train = random.sample(data_train, int(size_train))
sample_test = random.sample(data_test, int(size_test))

#normalize all words to lowercase
train_sent = [x.split(' ', 1)[1][:-1].lower() for x in sample_train]
test_sent = [x.split(' ', 1)[1][:-1].lower() for x in sample_test]


#Display #word per review distribution
words_per_review = list(map(lambda x: len(x.split()), train_sent))

sns.displot(words_per_review)
plt.xlabel('Word Length')
plt.ylabel('Number of Reviews')
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

train_df = pd.DataFrame(temp_train_data, columns = ('good_review', 'review_raw','preclean_len'))
test_df = pd.DataFrame(temp_test_data, columns = ('good_review', 'review_raw','preclean_len'))

train_df.good_review.value_counts()
    # #number of good and bad reviews are basically uniform
    # 0    45217
    # 1    44783
    # Name: good_review, dtype: int64
    
    
#cleanup irrelevant objects
del [size_train, size_test, words_per_review,train_label, test_label,
     review_length_train, review_length_test,
     temp_train_data, temp_test_data, data_train, data_test]

#compare words count by review type
sns.set_theme(style="darkgrid")
ax = sns.boxplot(x = 'good_review', y = 'preclean_len',
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
# Test
    # aa_trial = test_sent.copy()
    # aa_test = remove_punc(aa_trial)
    # ' '.join(aa_test[60])
    # ' '.join(aa_trial[60])

train_sent = remove_punc(train_sent)
test_sent = remove_punc(test_sent)

train_df['no_punc'] = train_sent
test_df['no_punc'] = test_sent

del [punc]

#replace digits
def digit_replace(corpus):
    
    output = []
    
    for docs in corpus:       
        
        temp_list = []
        for word in docs:
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


#identify and remove 'stopwords' (i.e. the, a)
stopwords = stopwords.words('english')
# keep_words = ['no', 'but', 'not']
# stopwords = [word for word in stopwords if word not in keep_words]

train_df['rmv_stopwords'] = train_df['no_punc'].apply(lambda x: [word for word in x if word not in stopwords])
test_df['rmv_stopwords'] = test_df['no_punc'].apply(lambda x: [word for word in x if word not in stopwords])

del [stopwords, keep_words]


#identify each token's part of speech
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

del wordnet

#Lemmatize
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

del wnl

#illustrate histogram of words
train_df['postclean_len'] = train_df['lemmatized'].apply(lambda x: len(x))
test_df['postclean_len'] = test_df['lemmatized'].apply(lambda x: len(x))

#compare before and after cleaned                     
ax = sns.displot(data = train_df, x = 'postclean_len', hue = 'good_review', kind = 'kde')

#violin plot of word length
ax = sns.catplot(data = train_df, y = 'postclean_len', palette = "Set2",x = 'good_review', kind = 'violin')

#list stats for postcleaned data
list_stats(train_df.loc[train_df['good_review']==1]['postclean_len'])
#good reviews --> mean: 41.73, median 36.0, max words: 175, min words: 6

list_stats(train_df.loc[train_df['good_review']==0]['postclean_len'])
#bad reviews -->mean: 45.42, median 41.0, max words: 163, min words: 7


#bag of words and rank them by review type

aa_temp = train_df.loc[89997:,['good_review','lemmatized']].copy()
aa_temp['sentences'] = aa_temp['lemmatized'].apply(lambda x: ' '.join(x))

train_df['sentences'] = train_df['lemmatized'].apply(lambda x: ' '.join(x))
train_df['sentences'] = train_df['lemmatized'].apply(lambda x: ' '.join(x))

#new df with cleaned words and sentences
df_train = train_df.loc[:,['good_review', 'postclean_len','lemmatized']]
df_test = test_df.loc[:,['good_review', 'postclean_len','lemmatized']]

#lemmatized sentences
df_train['lemm_sent'] = df_train['lemmatized'].apply(lambda x: ' '.join(x))
df_test['lemm_sent'] = df_test['lemmatized'].apply(lambda x: ' '.join(x))

#identify in english review type
df_train['review_type'] = df_train.good_review.map({0:'bad', 1:'good'})
df_test['review_type'] = df_test.good_review.map({0:'bad', 1:'good'})

df_train.review_type.value_counts()

#train_test_split data - simulate real world

final_train = df_train[:][['lemm_sent','good_review', 'review_type']]
final_test = df_test[:][['lemm_sent','good_review', 'review_type']]

final_train.columns = ['text', 'val', 'label']
final_test.columns = ['text', 'val', 'label']




#Vectorize dataset

# CountVectorizer data and labels top words
cv = CountVectorizer(min_df = 0.00, max_df=1.00, max_features=10000, ngram_range = (1,2))

cv.fit(final_train.text)

neg_matrix = cv.transform(final_train[final_train.val == 0].text)
neg_words = neg_matrix.sum(axis=0)
neg_words_list = [(word, neg_matrix[0, i]) for word, i in cv.vocabulary_.items()]
neg_word_freq = pd.DataFrame(list(sorted(neg_words_list, key = lambda x: x[1], reverse=True)),columns=['Term','Count'])
neg_cv = neg_word_freq.head(10)
neg_cv[:]['Count'] = neg_cv[:]["Count"].astype(int)


positive_matrix = cv.transform(final_train[final_train.val == 1].text)
positive_words = positive_matrix.sum(axis=0)
positive_words_list = [(word, positive_matrix[0, i]) for word, i in cv.vocabulary_.items()]
positive_word_freq = pd.DataFrame(list(sorted(positive_words_list, key = lambda x: x[1], reverse=True)),columns=['Term','Count'])
positive_cv = positive_word_freq.head(10)
positive_cv[:]['Count'] = positive_cv[:]["Count"].astype(int)


ax = sns.barplot(x = 'Count', y = 'Term', data = neg_cv)
ax.set_title('CountVec Negative')
ax.set(xlabel='Frequency', ylabel = 'Word')
plt.show()
ax.clear() 

ax = sns.barplot(x = 'Count', y = 'Term', data = positive_cv)
ax.set_title('CountVec Positive')
ax.set(xlabel='Frequency', ylabel = 'Word')
plt.show()
ax.clear() 
# TF-IDF data and labels top words
tfidf = TfidfVectorizer(min_df = 0.00, max_df=1.00, max_features=10000, ngram_range = (1,2))

tfidf.fit(final_train.text)

neg_matrix = tfidf.transform(final_train[final_train.val == 0].text)
neg_words = neg_matrix.sum(axis=0)
neg_words_list = [(word, neg_matrix[0, i]) for word, i in tfidf.vocabulary_.items()]
neg_word_freq = pd.DataFrame(list(sorted(neg_words_list, key = lambda x: x[1], reverse=True)),columns=['Term','Count'])
neg_tfidf = neg_word_freq.head(10)
neg_tfidf[:]['Count'] = neg_tfidf[:]["Count"].astype(int)


positive_matrix = tfidf.transform(final_train[final_train.val == 1].text)
positive_words = positive_matrix.sum(axis=0)
positive_words_list = [(word, positive_matrix[0, i]) for word, i in tfidf.vocabulary_.items()]
positive_word_freq = pd.DataFrame(list(sorted(positive_words_list, key = lambda x: x[1], reverse=True)),columns=['Term','Count'])
positive_tfidf = positive_word_freq.head(10)
positive_tfidf[:]['Count'] = positive_tfidf[:]["Count"].astype(int)


ax = sns.barplot(x = 'Count', y = 'Term', data = neg_tfidf)
ax.set_title('Tfidf Negative')
ax.set(xlabel='Importance', ylabel = 'Word')
plt.show()
ax.clear() 

ax = sns.barplot(x = 'Count', y = 'Term', data = positive_tfidf)
ax.set_title('Tfidf Positive')
ax.set(xlabel='Importance', ylabel = 'Word')
plt.show()
ax.clear()




#CountVectorizer
#test train split
X, y = final_train.text, final_train.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)

# Predictive Algorithms
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)


#A1)1 fold BernoulliNB
bnb = BernoulliNB()
%time bnb.fit(X_train_cv, y_train)

%time predict= bnb.predict(X_test_cv)
print()
print('Overall Accuracy Score: ',metrics.accuracy_score(y_test, predict))
print()
print('Confusion Matrix: ')
print(confusion_matrix(y_test, predict))
print()
print('Classification Report: ')
print(classification_report(y_test, predict))

        # Overall Accuracy Score:  0.8
        
        # Confusion Matrix: 
        # [[78 15]
        #  [25 82]]
        
        # Classification Report: 
        #               precision    recall  f1-score   support
        
        #          bad       0.76      0.84      0.80        93
        #         good       0.85      0.77      0.80       107
        
        #     accuracy                           0.80       200
        #    macro avg       0.80      0.80      0.80       200
        # weighted avg       0.80      0.80      0.80       200


#A2)1 fold Logistic Regression

lr = LogisticRegression()
%time lr.fit(X_train_cv, y_train)

%time predict= lr.predict(X_test_cv)
print()
print('Overall Accuracy Score: ',metrics.accuracy_score(y_test, predict))
print()
print('Confusion Matrix: ')
print(confusion_matrix(y_test, predict))
print()
print('Classification Report: ')
print(classification_report(y_test, predict))

        # Overall Accuracy Score:  0.78
        
        # Confusion Matrix: 
        # [[70 23]
        #  [21 86]]
        
        # Classification Report: 
        #               precision    recall  f1-score   support
        
        #          bad       0.77      0.75      0.76        93
        #         good       0.79      0.80      0.80       107
        
        #     accuracy                           0.78       200
        #    macro avg       0.78      0.78      0.78       200
        # weighted avg       0.78      0.78      0.78       200


#A3) BernoulliNB 10 fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=123)

scores = []
confusion = np.array([[0,0], [0,0]])

for train_indices, test_indices in kf.split(final_train):
    X_train = final_train.iloc[train_indices]['text']
    y_train = final_train.iloc[train_indices]['val']
    
    
    X_test = final_train.iloc[test_indices]['text']
    y_test = final_train.iloc[test_indices]['val']
     
    
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)
    
    bnb.fit(X_train_cv, y_train)
    predict = bnb.predict(X_test_cv)    


    confusion += confusion_matrix(y_test, predict)
    # print(confusion)
    score = f1_score(y_test, predict)
    scores.append(score)
print()
print('Overall Score:',round(sum(scores)/len(scores),2))
print()
print('Confusion Matrix: ')
print(confusion)
print()
print('Classification Report: ')
print(classification_report(y_test, predict))

        # Overall Score: 0.8
        
        # Confusion Matrix: 
        # [[335 171]
        #  [ 52 442]]
        
        # Classification Report: 
        #               precision    recall  f1-score   support
        
        #            0       0.93      0.58      0.72        48
        #            1       0.71      0.96      0.82        52
        
        #     accuracy                           0.78       100
        #    macro avg       0.82      0.77      0.77       100
        # weighted avg       0.82      0.78      0.77       100


#A4) LogisticRegression 10 fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=123)

scores = []
confusion = np.array([[0,0], [0,0]])


for train_indices, test_indices in kf.split(final_train):
    X_train = final_train.iloc[train_indices]['text']
    y_train = final_train.iloc[train_indices]['val']
    
    
    X_test = final_train.iloc[test_indices]['text']
    y_test = final_train.iloc[test_indices]['val']
     
    
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)
    
    lr.fit(X_train_cv, y_train)
    predict = lr.predict(X_test_cv)    


    confusion += confusion_matrix(y_test, predict)
    # print(confusion)
    score = f1_score(y_test, predict)
    scores.append(score)
    
print()
print('Overall Score:',round(sum(scores)/len(scores),2))
print()
print('Confusion Matrix: ')
print(confusion)
print()
print('Classification Report: ')
print(classification_report(y_test, predict))
        
        # Overall Score: 0.8
        
        # Confusion Matrix: 
        # [[398 108]
        #  [ 96 398]]
        
        # Classification Report: 
        #               precision    recall  f1-score   support
        
        #            0       0.77      0.71      0.74        48
        #            1       0.75      0.81      0.78        52
        
        #     accuracy                           0.76       100
        #    macro avg       0.76      0.76      0.76       100
        # weighted avg       0.76      0.76      0.76       100






#TFIDFVectorizer

# Vectorize data and labels
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#B1) fold BernoulliNB
bnb = BernoulliNB()
%time bnb.fit(X_train_tfidf, y_train)

%time predict= bnb.predict(X_test_tfidf)

print()
print('Overall Accuracy Score: ',metrics.accuracy_score(y_test, predict))
print()
print('Confusion Matrix: ')
print(confusion_matrix(y_test, predict))
print()
print('Classification Report: ')
print(classification_report(y_test, predict))        

        # Overall Accuracy Score:  0.8
        
        # Confusion Matrix: 
        # [[36 12]
        #  [ 8 44]]
        
        # Classification Report: 
        #               precision    recall  f1-score   support
        
        #            0       0.82      0.75      0.78        48
        #            1       0.79      0.85      0.81        52
        
        #     accuracy                           0.80       100
        #    macro avg       0.80      0.80      0.80       100
        # weighted avg       0.80      0.80      0.80       100

#B2)1 fold LogisticRegression
lr = LogisticRegression()
%time lr.fit(X_train_tfidf, y_train)

%time predict= lr.predict(X_test_tfidf)

print()
print('Overall Accuracy Score: ',metrics.accuracy_score(y_test, predict))
print()
print('Confusion Matrix: ')
print(confusion_matrix(y_test, predict))
print()
print('Classification Report: ')
print(classification_report(y_test, predict))     
       
         # Overall Accuracy Score:  0.79
        
        # Confusion Matrix: 
        # [[39  9]
        #  [12 40]]
        
        # Classification Report: 
        #               precision    recall  f1-score   support
        
        #            0       0.76      0.81      0.79        48
        #            1       0.82      0.77      0.79        52
        
        #     accuracy                           0.79       100
        #    macro avg       0.79      0.79      0.79       100
        # weighted avg       0.79      0.79      0.79       100

#B3) BernoulliNB 10 fold cross validation

kf = KFold(n_splits=10, shuffle=True, random_state=123)

scores = []
confusion = np.array([[0,0], [0,0]])

for train_indices, test_indices in kf.split(final_train):
    X_train = final_train.iloc[train_indices]['text']
    y_train = final_train.iloc[train_indices]['val']
    
    
    X_test = final_train.iloc[test_indices]['text']
    y_test = final_train.iloc[test_indices]['val']
     
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    bnb.fit(X_train_tfidf, y_train)
    predict = bnb.predict(X_test_tfidf)    


    confusion += confusion_matrix(y_test, predict)
    # print(confusion)
    score = f1_score(y_test, predict)
    scores.append(score)
print()
print('Overall Score:',round(sum(scores)/len(scores),2))
print()
print('Confusion Matrix: ')
print(confusion)
print()
print('Classification Report: ')
print(classification_report(y_test, predict))

        # Overall Score: 0.79
        
        # Confusion Matrix: 
        # [[377 129]
        #  [ 86 408]]
        
        # Classification Report: 
        #               precision    recall  f1-score   support
        
        #            0       0.82      0.75      0.78        48
        #            1       0.79      0.85      0.81        52
        
        #     accuracy                           0.80       100
        #    macro avg       0.80      0.80      0.80       100
        # weighted avg       0.80      0.80      0.80       100


        # Confusion Matrix:
        # [[377 129]
        #  [ 86 408]]
        # Score: 0.79
        # Classification Report:
        #               precision    recall  f1-score   support
        
        #            0       0.82      0.75      0.78        48
        #            1       0.79      0.85      0.81        52
        
        #     accuracy                           0.80       100
        #    macro avg       0.80      0.80      0.80       100
        # weighted avg       0.80      0.80      0.80       100



#B4) LogisticRegression 10 fold cross validation

kf = KFold(n_splits=10, shuffle=True, random_state=123)

scores = []
confusion = np.array([[0,0], [0,0]])

for train_indices, test_indices in kf.split(final_train):
    X_train = final_train.iloc[train_indices]['text']
    y_train = final_train.iloc[train_indices]['val']
    
    
    X_test = final_train.iloc[test_indices]['text']
    y_test = final_train.iloc[test_indices]['val']
     
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    lr.fit(X_train_tfidf, y_train)
    predict = lr.predict(X_test_tfidf)    


    confusion += confusion_matrix(y_test, predict)
    # print(confusion)
    score = f1_score(y_test, predict)
    scores.append(score)
print()
print('Overall Score:',round(sum(scores)/len(scores),2))
print()
print('Confusion Matrix: ')
print(confusion)
print()
print('Classification Report: ')
print(classification_report(y_test, predict))

        # Overall Score: 0.8
        
        # Confusion Matrix: 
        # [[421  85]
        #  [110 384]]
        
        # Classification Report: 
        #               precision    recall  f1-score   support
        
        #            0       0.76      0.81      0.79        48
        #            1       0.82      0.77      0.79        52
        
        #     accuracy                           0.79       100
        #    macro avg       0.79      0.79      0.79       100
        # weighted avg       0.79      0.79      0.79       100




# roadap:
#Still need to do grid search, 
# update y-pred / test (in kfold cv)
# update visualizations (xlabel, y label)
# display top words based on review type (cv vs tf-idf)
    # How can I display most relevant words and not just the most frequent?
#update url pattern
#update punctuation removal


#how to update so that 'count' is as integars rather than float
    # ax = sns.barplot(x = 'Count', y = 'Term', data = neg_df)