from datetime import time
from pprint import pprint

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=40)
#print(twenty_train.data)
#print(type(twenty_train.data))
#print(twenty_train.filenames)
#print(twenty_train.target) #array [1 1 3 ..., 2 2 2]
#print(twenty_train.target_names) #array ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
#print(twenty_train.target_names[twenty_train.target[0]])
#It is possible to get back the category names as follows:
#for t in twenty_train.target[:100]:
#    print(t)
#    print(twenty_train.target_names[t])


#CountVect
# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)
# #print(X_train_counts)
# print(count_vect.get_feature_names())
# print(X_train_counts.toarray())
# #count_vect.vocabulary_.get(u'algorithm')
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.toarray())


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
tfidf_vectorizer = TfidfVectorizer()
svd = TruncatedSVD(n_components=50, random_state=25)
# Pipeline Vectorizer and SVD
svd_transformer = Pipeline([('tfidf', tfidf_vectorizer),
                            ('svd', svd)])
svd_matrix = svd_transformer.fit_transform(twenty_train.data)
#print("Pipeline\n",svd_matrix)
print(svd_matrix.shape)

from Functions import newsGroupClassifier
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=40)
svm = SGDClassifier(loss='squared_hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=40) # 40 > 82
svc = SVC(kernel='linear',decision_function_shape='ovo',probability=False,random_state=40) # 40 > 57
svcl = LinearSVC(loss='squared_hinge', penalty='l2', max_iter=5, random_state=40)

#naivebayes = MultinomialNB()
naivebayes = GaussianNB()
logistic = LogisticRegression(penalty='l2', max_iter=5, random_state=40)

#newsGroupClassifier(svd_transformer,naivebayes,twenty_train,twenty_test)
newsGroupClassifier(svd_transformer,svc,twenty_train,twenty_test)
#newsGroupClassifier(svd_transformer,svcl,twenty_train,twenty_test)
#newsGroupClassifier(svd_transformer,logistic,twenty_train,twenty_test)
import numpy as np
from sklearn import metrics
# Grid Search
# define a pipeline
pipeline = Pipeline([
    ('vect', svd_transformer),
    ('svmobj', svc),
])
print(pipeline.get_params().keys())

parameters = {
    #'svmobj__alpha': (0.00001, 0.000001),
    #'svmobj__penalty': ('l2', 'elasticnet'),
    'svmobj__gamma': [1e-3, 1e3]
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
    print("Performing grid search...")
    #print("pipeline:", [name for name, _ in pipeline.steps])
    #print("parameters:")
    #  pprint(parameters)
    t0 = time()
    grid_search.fit(twenty_train.data, twenty_train.target)
    predicted = grid_search.predict(twenty_test.data)
    print(np.mean(predicted == twenty_test.target))
    print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
    metrics.confusion_matrix(twenty_test.target, predicted)
    #print("done in %0.3f" % (time() - t0))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))