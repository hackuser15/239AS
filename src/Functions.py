import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn import cross_validation

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