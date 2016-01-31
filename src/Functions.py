import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn import cross_validation, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import Plots

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
    print('%s - Averaged RMSE with CV %.4f ' % (label, np.mean(np.sqrt(-scores))))
    print('%s - Best RMSE with CV is: %.4f' % (label,bestRMSE(scores)))
    return predicted

def callCrossValPoly(obj,X,y,numberoffolds,degree,label):
    predicted = cross_val_predict(obj, X, y, cv=numberoffolds)
    scores = cross_val_score(obj, X, y, cv=numberoffolds, scoring='mean_squared_error')
    print('%s - The best RMSE obtained with CV for degree %d is: %.4f' % (label,degree,bestRMSE(scores)))
    return bestRMSE(scores)

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
        print("Root Mean Squared Error (Polynomial) for degree %d: %.4f" % (degree,rmse))
        #print('Variance score (Polynomial): %.4f' % model.score(X_test, y_test))
    print('The best RMSE obtained is: %.4f for Degree: %d' % (minRMSE(rmseList),rmseList.index(minRMSE(rmseList))+1))
    Plots.scatterPlot(degreeList, rmseList, 'Degree_of_Polynomial', 'RMSE', 'Title', 'green', 'polyreg')

def polynomialRegressionNew(obj,X_train,y_train,X_test,y_test, degree, label):
    rmseList = []
    degreeList = []
    for deg in range(1,degree+1):
        poly = PolynomialFeatures(degree=deg)
        X_ = poly.fit_transform(X_train)
        predict_ = poly.fit_transform(X_test)
        obj.fit(X_, y_train)
        pred=obj.predict(predict_)
        rmse=rootMeanSquareError(pred, y_test)
        degreeList.append(degree)
        rmseList.append(rmse)
        print("Root Mean Squared Error: (Polynomial): %.4f" % rmse)
       # print('Variance score (Polynomial): %.4f' % model.score(X_test, y_test))
    print('The best RMSE obtained is: %.4f for Degree: %d' % (minRMSE(rmseList),rmseList.index(minRMSE(rmseList))+1))
    Plots.scatterPlot(degreeList, rmseList, 'Degree_of_Polynomial', 'RMSE', 'Title', 'green', 'polyreg')

def polynomialRegressionCV(obj,X,y,numberoffolds,degree,label):
    rmseList = []
    degreeList = []
    for degree in range(1,degree+1):
        model = make_pipeline(PolynomialFeatures(degree), obj)
        rmse = callCrossValPoly(model, X, y, numberoffolds, degree, 'Linear Regression Poly')
        degreeList.append(degree)
        rmseList.append(rmse)
    print('The best RMSE obtained is: %.4f for Degree: %d' % (minRMSE(rmseList),rmseList.index(minRMSE(rmseList))+1))
    Plots.scatterPlot(degreeList, rmseList, 'Degree_of_Polynomial', 'RMSE', 'Title', 'red', 'polyregCV')

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
