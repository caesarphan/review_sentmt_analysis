#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:11:01 2021

@author: caesarphan
"""
#123
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
import re
import spacy
import bz2

f = open("train.ft.txt", "r")

train_file = bz2.BZ2File('/Users/caesarphan/Documents/Projects/amazon_reviews/train.ft.txt.bz2')
test_file = bz2.BZ2File('/Users/caesarphan/Documents/Projects/amazon_reviews/test.ft.txt.bz2')

train_file = [x.decode('utf-8') for x in train_file]
test_file = [x.decode('utf-8') for x in test_file]


train_file[0][0:10]


temp_range = np.arange(1,30)

def review_type(x):
    return 1 if re.match("__label__2", x) is not None else 0

train_label = [0 if x[:10] == '__label__1' else 1 for x in train_file]

token_list = []
for i in temp_range:
    tk = RegexpTokenizer('\s+', gaps = True)
    x = tk.tokenize(train_file[i])
    token_list.append(x)
    
token_list[0][0].pos_

