import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import metrics

def newsGroupClassifier(svd_transformer,classifier_obj,twenty_train,twenty_test):
    obj = Pipeline([('vect', svd_transformer),
                    ('clf', classifier_obj),
                   ])
    obj = obj.fit(twenty_train.data, twenty_train.target)
    predicted = obj.predict(twenty_test.data)
    print(np.mean(predicted == twenty_test.target))
    print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
    print(metrics.confusion_matrix(twenty_test.target, predicted))