import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score

def meanSquareError(x, y):
    return np.mean((x - y) ** 2)

def printFeatureCoefficients(x, y):
    zipped=list(zip(x, y))
    print(pd.DataFrame(data=zipped,columns=['Features','Coefficents']))

def crossValScoreRMSE(scores):
    MSE = -scores
    return np.sqrt(np.mean(MSE))

def callClassifier(obj,X_train,y_train,X_test,y_test,feature_cols, label):
    obj.fit(X_train, y_train)
    printFeatureCoefficients(feature_cols, obj.coef_)
    obj.predict(X_test)
    print('%s - Residual sum of squares: %.2f' % (label, meanSquareError(obj.predict(X_test), y_test)))
    print('%s - Variance score: %.2f' % (label, obj.score(X_test, y_test)))

def callCrossVal(obj,X,y,numberoffolds,label):
    predicted = cross_val_predict(obj, X, y, cv=numberoffolds)
    scores = cross_val_score(obj, X, y, cv=numberoffolds, scoring='mean_squared_error')
    #print(scores)
    print('The best RMSE obtained with CV is: %.2f' % bestRMSE(scores))
    print('%s - Mean Squared Error with CV %.2f ' % (label, np.mean(-scores)))
    print('%s - Root Mean Squared Error with CV %.2f ' % (label, np.sqrt(np.mean(-scores))))

def bestRMSE(scores):
    MSE = np.sqrt(-scores)
    return np.amin(MSE)