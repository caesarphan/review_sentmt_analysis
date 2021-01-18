# review_sentmt_analysis
NLP amazon sentiment analysis

data: dataset 3.6 million observations. Dataset stored locally in 'documents/proects' directory

labels:
0 --> negative review
1 --> positive review

Vectorizing using CountVectorizer vs TFIDF Vectorizer, TfidfVectorizer consistently had higher overall accuracy

Bernoulli Naive Bayes vs Logistic Regression:
Based on 4-fold gridsearch, we established the optimal parameters for both models.
Based on these parameters, we trained the training data set then fitten the testing.

Conclusion:
ASSUMPTIONS: sampled datapoints (~1/40th dataset) representative of entire dataset
Overall, Logistic regression (via tfidf) consistenly outperformed BernoulliNB
Against the testing dataset, Logistic Regression (88.79%) was 2.7% better at classifying reviews than BernoulliNB (85.11%)
