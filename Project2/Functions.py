import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import metrics
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import collections

from Project2.Plots import plotROC


def newsGroupClassifier(svd_transformer,classifier_obj,twenty_train,twenty_test,name):
    obj = Pipeline([('vect', svd_transformer),
                    ('clf', classifier_obj),
                   ])
    obj = obj.fit(twenty_train.data, twenty_train.target)
    predicted = obj.predict(twenty_test.data)
    preds = obj.predict_proba(twenty_test.data)
    calcPrintResults(twenty_test,predicted,name)
    fpr, tpr, thresholds = metrics.roc_curve(twenty_test.target, preds[:,1])
    plotROC(fpr,tpr,name)

def newsGroupMultiClassifier(svd_transformer,classifier_obj,twenty_train,twenty_test,name):
    obj = Pipeline([('vect', svd_transformer),
                    ('clf', classifier_obj),
                   ])
    obj = obj.fit(twenty_train.data, twenty_train.target)
    predicted = obj.predict(twenty_test.data)
    calcPrintResults(twenty_test,predicted,name)

def calcPrintResults(twenty_test,predicted,name):
    print("Classification report for classifier %s:\n%s" % (name,metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names)))
    print("Confusion matrix:\n%s\n" % metrics.confusion_matrix(twenty_test.target, predicted))


#Function to remove punctuation,stop words and lemmatizing
def cleanDoc(doc):
    stop = stopwords.words('english')
    lmtzr = WordNetLemmatizer()
    #print("Original\n",doc)
    cleaned =""
    for i in word_tokenize(doc.lower()):
        if i not in stop:
            root = lmtzr.lemmatize(i)
            if root not in string.punctuation:
                cleaned = cleaned+ " "+root
    return cleaned

def getAllDocsForClass(train_all, index):
    doclist = []
    for i in range(0, len(train_all.data)):
        if(train_all.target[i] == index):
            doclist.append(i)
    return doclist

def printTop10(final_terms, count_vect):
    for i in final_terms:
        map = i
        d = collections.Counter(i)
        d.most_common()
        for k, v in d.most_common(10):
            print(count_vect.get_feature_names()[k]," ", v)
            print("-"*100)