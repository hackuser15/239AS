import sklearn
from sklearn import metrics

from nltk.chunk.named_entity import shape
from scipy.odr import models
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import string

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import math

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from datetime import time
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC

import Functions
from Project2.Functions import calcPrintResults, cleanDoc, cleaned_data, cleaned_datawithDecode

########################## END OF C ####################################
#Applying LSI to the TF-IDF matrix to reduce to 50 features
train_data = sklearn.datasets.load_files("20news-bydate-train")
test_data = sklearn.datasets.load_files("20news-bydate-test")
train_data=cleaned_datawithDecode(train_data)
test_data=cleaned_datawithDecode(test_data)

vectorizer = TfidfVectorizer()
svd = TruncatedSVD(n_components=50, n_iter=5, random_state=25)
svd_transformer = Pipeline([('tfidf', vectorizer),
                            ('svd', svd)])
svd_matrix = svd_transformer.fit_transform(train_data.data)

########################## END OF D ####################################
#Support Vector Machine
print("Support Vector Machine Analysis")
svc = SVC(kernel='linear', probability=True, random_state=40) #decision_function_shape='ovo',max_iter=5
Functions.newsGroupClassifier(svd_transformer,svc,train_data,test_data,'Support Vector Machine')

########################## END OF E ####################################
#Soft margin SVM

# Grid Search
# print("Soft Margin SVM Analysis")
# pipeline = Pipeline([
#     ('vect', svd_transformer),
#     ('svmobj', svc),
# ])
# parameters = {
#     'svmobj__gamma': np.logspace(-3, 3, num=7, base= np.exp(1))
# }
# if __name__ == "__main__":
#         grid_search = GridSearchCV(pipeline, parameters, cv=5) #n_jobs=1
#         grid_search.fit(train_data.data, train_data.target)
#         predicted = grid_search.predict(test_data.data)
#         calcPrintResults(test_data,predicted,'Soft Margin SVM')
#         print("Best parameters set:")
#         best_parameters = grid_search.best_estimator_.get_params()
#         for param_name in sorted(parameters.keys()):
#             print("\t%s: %r" % (param_name, best_parameters[param_name]))

########################## END OF F ####################################
#Naive Bayes
print("Naive Bayes Analysis")
naivebayes = GaussianNB()
Functions.newsGroupClassifier(svd_transformer,naivebayes,train_data,test_data,'Naive Bayes')

########################## END OF G ####################################
#Logistic Regression
print("Logistic Regression Analysis")
logistic = LogisticRegression(penalty='l2', max_iter=5, random_state=40)
Functions.newsGroupClassifier(svd_transformer,logistic,train_data,test_data,'Logistic Regression')

########################## END OF H ####################################
#MultiClass Classification
categories = ['comp.sys.ibm.pc.hardware' , 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
train_data = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=40)
test_data = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=40)
train_data=cleaned_data(train_data)
test_data=cleaned_data(test_data)
print("Naive Bayes Multi Class Analysis")
naivebayes = GaussianNB()
Functions.newsGroupMultiClassifier(svd_transformer,naivebayes,train_data,test_data,'Naive Bayes MultiClass')
print("OneVsRestClassifier Analysis")
obj = OneVsRestClassifier(svc)
Functions.newsGroupMultiClassifier(svd_transformer, obj, train_data, test_data, 'OneVsRestClassifier')
print("OneVsOneClassifier Analysis")
svc = SVC(kernel='linear',class_weight='balanced',probability=True,random_state=40)
obj = OneVsOneClassifier(svc)
Functions.newsGroupMultiClassifier(svd_transformer, obj, train_data, test_data, 'OneVsOneClassifier')

########################## END OF I ####################################