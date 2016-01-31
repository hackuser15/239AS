import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Plots
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def rootMeanSquareError(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def avgRMSE(scores):
    return np.mean(np.sqrt(-scores))

def bestRMSE(scores):
    MSE = np.sqrt(-scores)
    return np.amin(MSE)

def minRMSE(rmseList):
    return np.amin(rmseList)

def crossValScoreRMSE(scores):
    MSE = -scores
    return np.sqrt(np.mean(MSE))

def printFeatureCoefficients(x, y):
    zipped=list(zip(x, y))
    print(pd.DataFrame(data=zipped,columns=['Features','Coefficents']))

def callClassifier(obj,X_train,y_train,X_test,y_test, label):
    obj.fit(X_train, y_train)
    pred=obj.predict(X_test)
    print('%s - Root Mean Squared Error: %.4f' % (label, rootMeanSquareError(pred, y_test)))
    #print('%s - Variance score: %.4f' % (label, obj.score(X_test, y_test)))
    return pred

def callClassifierFeatures(obj,X_train,y_train,X_test,y_test,feature_cols, label):
    obj.fit(X_train, y_train)
    printFeatureCoefficients(feature_cols, obj.coef_)
    pred=obj.predict(X_test)
    print('%s - Root Mean Squared Error: %.4f' % (label, rootMeanSquareError(pred, y_test)))
    #print('%s - Variance score: %.4f' % (label, obj.score(X_test, y_test)))
    return pred

def callCrossVal(obj,X,y,numberoffolds,label):
    predicted = cross_val_predict(obj, X, y, cv=numberoffolds)
    scores = cross_val_score(obj, X, y, cv=numberoffolds, scoring='mean_squared_error')
    print('%s - Averaged RMSE with CV: %.4f ' % (label, np.mean(np.sqrt(-scores))))
    print('%s - Best RMSE with CV: %.4f' % (label,bestRMSE(scores)))
    return predicted

def polynomialRegression(obj,X_train,y_train,X_test,y_test, degree, label):
    rmseList = []
    degreeList = []
    for degree in range(1,degree+1):
        model = make_pipeline(PolynomialFeatures(degree), obj)
        model.fit(X_train, y_train)
        pred=model.predict(X_test)
        rmse=rootMeanSquareError(pred, y_test)
        degreeList.append(degree)
        rmseList.append(rmse)
        print("%s (Polynomial) - Root Mean Squared Error for degree %d: %.4f" % (label,degree,rmse))
        #print('Variance score (Polynomial): %.4f' % model.score(X_test, y_test))
    print('%s (Polynomial) - The best RMSE obtained is %.4f for degree: %d' % (label,minRMSE(rmseList),rmseList.index(minRMSE(rmseList))+1))
    Plots.scatterPlot(degreeList, rmseList, 'Degree_of_Polynomial', 'RMSE', 'Degree VS RMSE', 'green', 'PolyRegression'+label)

def polynomialRegressionCV(obj,X,y,numberoffolds,degree,label):
    rmseList = []
    degreeList = []
    for degree in range(1,degree+1):
        model = make_pipeline(PolynomialFeatures(degree), obj)
        rmse = callCrossValPoly(model, X, y, numberoffolds, degree, label)
        degreeList.append(degree)
        rmseList.append(rmse)
    #print('%s (Polynomial) - The best RMSE obtained is: %.4f for Degree: %d' % (label,minRMSE(rmseList),rmseList.index(minRMSE(rmseList))+1))
    Plots.scatterPlot(degreeList, rmseList, 'Degree_of_Polynomial', 'RMSE', 'Degree VS RMSE', 'red', 'PolyRegressionCV'+label)

def callCrossValPoly(obj,X,y,numberoffolds,degree,label):
    predicted = cross_val_predict(obj, X, y, cv=numberoffolds)
    scores = cross_val_score(obj, X, y, cv=numberoffolds, scoring='mean_squared_error')
    AvgRMSE = np.mean(np.sqrt(-scores))
    print('%s (Polynomial) - Averaged RMSE with CV: %.4f for degree: %d' % (label,AvgRMSE,degree))
    return AvgRMSE

def plotWorkFlow(data, num, label):
    workflow = data[data['Work-Flow-ID=work_flow_' + str(num)] == 1]
    num_operation = workflow['Size of Backup (GB)'].size
    plt.scatter(np.linspace(1, num_operation, num_operation), workflow['Size of Backup (GB)'])
    plt.xlabel('Operations')
    plt.ylabel('Size of Backup (GB)')
    plt.title('Workflow '+ str(num)+ ' ' + label)
    plt.savefig('Workflow_' + str(num) + '_' + label + '.png')
    plt.clf()

def fitWorkFlow(model, X, y, num):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=3)
    model.fit(X_train, y_train)
    print('Linear Regression workflow %d- RMSE: %.4f' % (num, np.sqrt(np.sum((model.predict(X_test) - y_test) ** 2)/y_test.size)))
