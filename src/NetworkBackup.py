#!/usr/bin/env python -W ignore::DeprecationWarning
import pandas
from OneHotEncode import one_hot_dataframe
import matplotlib.pyplot as plt
import Functions
import Plots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#from sklearn.neural_network import MLPRegressor
from sklearn import cross_validation
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.grid_search import RandomizedSearchCV,GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

network_data = pandas.read_csv('network_backup_dataset.csv')

one_hot_data, _, _ = one_hot_dataframe(network_data, ['Day of Week', 'Work-Flow-ID','File Name'], replace=True)

one_hot_subset = one_hot_data[one_hot_data['Week #'] <= 3]

for num in range(0,5):
    Functions.plotWorkFlow(one_hot_subset,num, 'actual')

feature_cols = [col for col in one_hot_data.columns if col not in ['Size of Backup (GB)']]
X = one_hot_data[feature_cols]
y = one_hot_data['Size of Backup (GB)']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=3)

model = LinearRegression()
Functions.callClassifier(model, X_train, y_train, X_test, y_test, feature_cols, 'Linear Regression')
#model.fit(X_train, y_train)
#print('Linear Regression - RMSE: %.4f' % (np.sqrt(np.sum((model.predict(X_test) - y_test) ** 2)/y_test.size)))

pred = model.predict(X_test)
Plots.scatterPlot(pred, y_test, 'Fitted','Actual','Fitted VS Actual','green','NetBkpLRFitvsActual')
# plt.scatter(model.predict(X_test), y_test,  color='green')
# plt.xlabel('Fitted')
# plt.ylabel('Actual')
# plt.title('Fitted VS Actual')
# #plt.show()
# plt.savefig('NetBkpLRFitvsActual.png')
# plt.clf()

Plots.residualPlot(pred, pred - y_test, 'Fitted','Resuduals','Fitted VS Residual','green','NetBkpLRFitvsResidual')
# plt.scatter(model.predict(X_test), (model.predict(X_test) - y_test),  color='blue')
# #plt.hlines(y=0,xmin=0,xmax=50)
# plt.xlabel('Fitted')
# plt.ylabel('Residual')
# plt.title('Fitted VS Residual')
# #plt.show()
# plt.savefig('NetBkpLRFitvsResidual.png')
# plt.clf()

Functions.callCrossVal(model, X, y, 10, 'Linear Regression')
#####################################################################################
model = RandomForestRegressor(n_estimators=20, max_depth=4, max_features='auto')
model.fit(X_train, y_train)
print('Random Forests Initial - RMSE: %.4f' % (np.sqrt(np.sum((model.predict(X_test) - y_test) ** 2)/y_test.size)))

# clf = RandomForestRegressor()
# param_dist = {"n_estimators":sp_randint(1, 100),
#               "max_depth": sp_randint(1, 10),
#               "max_features": sp_randint(1, 45),
#               "min_samples_split": sp_randint(1, 11),
#               "min_samples_leaf": sp_randint(1, 11),
#               "bootstrap": [True, False]}
#
# #run randomized search
# n_iter_search = 20
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search)

# random_search.fit(X, y)
# print(random_search.best_params_)
# print(random_search.best_score_)

# param_grid = {"n_estimators":list(range(1, 101, 10)),
#               "max_depth": list(range(1, 10, 5)),
#               "max_features": [1, 10, 19, 28, 37, 45],
#               "min_samples_split": list(range(1, 10, 5)),
#               "min_samples_leaf": list(range(1, 10, 5)),
#               "bootstrap": [True, False]}
#
#run grid search
# grid_search = GridSearchCV(clf, param_grid=param_grid)
# grid_search.fit(X, y)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

#model = RandomForestRegressor(n_estimators=79, min_samples_split= 8, max_features= 40, min_samples_leaf= 5, bootstrap= True, max_depth= None)
model = RandomForestRegressor(n_estimators=50, min_samples_split= 4, max_features= 41, min_samples_leaf= 4, bootstrap= True, max_depth= 9)
model.fit(X_train, y_train)
print('Random Forests after tuning - RMSE: %.4f' % (np.sqrt(np.sum((model.predict(X_test) - y_test) ** 2)/y_test.size)))

pred = model.predict(X_test)
Plots.scatterPlot(pred, y_test, 'Fitted','Actual','Fitted VS Actual RF','green','RandomForestActualvsFit')
# plt.scatter(model.predict(X_test), y_test,  color='green')
# plt.xlabel('Fitted')
# plt.ylabel('Actual')
# plt.title('Fitted VS Actual RF')
# #plt.show()
# plt.savefig('RandomForestActualvsFit.png')
# plt.clf()

Plots.residualPlot(pred, pred - y_test, 'Fitted','Resuduals','Fitted VS Residual','green','RandomForestFitvsResidual')

pred = X_test.copy()
pred['Size of Backup (GB)'] = pandas.Series(model.predict(X_test), index=pred.index)
predicted_subset = pred[pred['Week #'] <= 3]
for num in range(0,5):
    Functions.plotWorkFlow(predicted_subset, num, 'predicted')
######################################################################################
#model = MLPRegressor()
#Functions.callClassifier(model, X_train, y_train, X_test, y_test, feature_cols, 'Neural Networks')
ds = SupervisedDataSet( 45, 1 )
ds.setField( 'input', X_train )
y_train_nn = y_train.copy().reshape( -1, 1 )
ds.setField( 'target', y_train_nn )

hidden_size = 1   # arbitrarily chosen

net = buildNetwork( 45, hidden_size, 1, bias = True )
trainer = BackpropTrainer( net, ds )

ds_test = SupervisedDataSet( 45, 1 )
ds_test.setField( 'input', X_test )
y_test_nn = y_test.copy().reshape( -1, 1 )
ds_test.setField( 'target', y_test_nn )

#trainer.trainOnDataset( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )
trainer.trainOnDataset(ds, 2)
p = net.activateOnDataset( ds_test )
print('Neural Network - RMSE: %.4f' % (np.sqrt(np.sum((p - y_test_nn) ** 2)/y_test.size)))
#####################################################################################

for num in range(0,5):
    data_workflow =  one_hot_data[one_hot_data['Work-Flow-ID=work_flow_'+str(num)] == 1]
    X = data_workflow[feature_cols]
    y = data_workflow['Size of Backup (GB)']
    Functions.fitWorkFlow(LinearRegression(), X, y, num)

Functions.polynomialRegression(LinearRegression(), X_train, y_train, X_test, y_test,6,'NetworkBackupPoly')
# Functions.polynomialRegressionNew(LinearRegression(), X_train, y_train, X_test, y_test, 6, 'NetworkBackupPoly')
# rmseList = []
# for degree in [1, 2, 3, 4, 5]:
#     model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
#     model.fit(X_train, y_train)
#     rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))
#     rmseList.append(rmse)
#     print('Polynomial Reg degree %d - RMSE: %.4f' % (degree, rmse))

#Plots.scatterPlot(np.linspace(1,5,5), rmseList, 'Degree','RMSE','RMSE vs Degree Fixed','green','RMSEvsDegreeFixed')
# plt.scatter(np.linspace(1,5,5), rmseList,  color='green')
# plt.xlabel('Degree')
# plt.ylabel('RMSE')
# plt.title('RMSE vs Degree Fixed')
# #plt.show()
# plt.savefig('RMSEvsDegreeFixed.png')
# plt.clf()

Functions.polynomialRegressionCV(LinearRegression(), X, y, 10,6, 'NetworkBackupPolyCV')
# rmseList = []
# for degree in [1, 2, 3, 4, 5]:
#     model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
#     predicted = cross_val_predict(model, X, y, cv=10)
#     rmse = np.sqrt(np.mean((predicted - y) ** 2))
#     rmseList.append(rmse)
#     print('Polynomial Reg degree %d - RMSE: %.4f' % (degree, rmse))

# Plots.scatterPlot(np.linspace(1,5,5), rmseList, 'Degree','RMSE','RMSE vs Degree CV','green','RMSEvsDegreeCV')
# plt.scatter(np.linspace(1,5,5), rmseList,  color='green')
# plt.xlabel('Degree')
# plt.ylabel('RMSE')
# plt.title('RMSE vs Degree CV')
# #plt.show()
# plt.savefig('RMSEvsDegreeCV.png')
# plt.clf()