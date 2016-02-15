from nltk.chunk.named_entity import shape
from scipy.odr import models
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import string
import Plots
import Functions
import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import math

import gensim
from gensim import corpora, models, utils
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from datetime import time
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC


# Fetching data for 8 specific categories:
computer_count = 0
recreational_count = 0
categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

train_1 = fetch_20newsgroups(subset='train', categories = categories, shuffle = True, random_state=42)
length = len(train_1.data)
category_count = {}
#


for i in range(0,length):
    category = train_1.target_names[train_1.target[i]]
    if(category in category_count):
        category_count[category] = category_count[category] + 1
    else:
        category_count[category] = 1
    if("comp" in category):
        computer_count = computer_count + 1
    else:
        recreational_count = recreational_count + 1


print("Computer category count= "+str(computer_count))
print("Recreational category count= "+str(recreational_count))
Plots.barPlot(category_count)

########################## END OF A ####################################
train_all = fetch_20newsgroups(subset='train', shuffle = True, random_state=42)
no_of_docs = len(train_all.data)

#Clean all documents
for i in range(0, 3):                   #Change this to no_of_docs
    cleaned_doc = Functions.cleanDoc(train_all.data[i])
    train_all.data[i] = cleaned_doc


#Convert to TfIdfVector
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(train_all.data)

print("Term Count = ",vectors.shape[1])

########################## END OF B ####################################
categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
indices = []
for i in categories:
    indices.append(train_all.target_names.index(i))

#Generating term frequency vector for all documents
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_all.data)
#print(count_vect.get_feature_names()) Will print names of all final terms

#Calculating TC-ICF
final_word_index = []
final_terms = []
for i in range(0, len(indices)):
    print("Searching for category ,",indices[i])
    doc_class_list = Functions.getAllDocsForClass(train_all, indices[i])
    list_count = [0]* X_train_counts.shape[1]
    significant_terms = {}
    #Process a single document and increment appropriate word counts
    for j in range(0, len(doc_class_list)):
        curr_doc = doc_class_list[j]
        non_zero_word_indices = X_train_counts[curr_doc, :].nonzero()[1]
        for word_index in non_zero_word_indices:
            freq = X_train_counts[curr_doc, word_index]
            if(word_index in significant_terms):
                significant_terms[word_index] = significant_terms[word_index] + freq;
            else:
                significant_terms[word_index] = freq
            list_count[word_index] = list_count[word_index] + freq

    final_word_index.append(max(list_count))
    #For each significant term in this class, find no. of classes which contain this term
    terms = {}
    for term,freq in significant_terms.items():
        contained_classes = []
        contained_doc_numbers = X_train_counts[:, term].nonzero()[0]
        for z in contained_doc_numbers:
            class_number = train_all.target[z]
            if(class_number not in contained_classes):
                contained_classes.append(class_number)
        score = 0.5+((0.5*freq/final_word_index[i])*(math.log10(20/len(contained_classes))))
        terms[term] = score
    final_terms.append(terms)


#Print top 10 terms in each of the 4 categories

Functions.printTop10(final_terms, count_vect)

########################## END OF C ####################################

#Applying LSI to the TF-IDF matrix to reduce to 50 features

svd = TruncatedSVD(n_components=50, random_state=25)
# Pipeline Vectorizer and SVD
svd_transformer = Pipeline([('tfidf', vectorizer),
                            ('svd', svd)])
svd_matrix = svd_transformer.fit_transform(train_all.data)
#print("Pipeline\n",svd_matrix)
print(svd_matrix.shape)

########################## END OF D ####################################
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=40)


#Linear Support Vector Machine
svcl = LinearSVC(loss='squared_hinge', penalty='l2', max_iter=5, random_state=40)

Functions.newsGroupClassifier(svd_transformer,svcl,train_all,twenty_test)

########################## END OF E ####################################

#Soft margin SVM



########################## END OF F ####################################

#Naive Bayes
#naivebayes = MultinomialNB()
naivebayes = GaussianNB()
Functions.newsGroupClassifier(svd_transformer,naivebayes,train_all,twenty_test)

########################## END OF G ####################################

#Logistic Regression
logistic = LogisticRegression(penalty='l2', max_iter=5, random_state=40)
Functions.newsGroupClassifier(svd_transformer,logistic,train_all,twenty_test)

########################## END OF H ####################################





