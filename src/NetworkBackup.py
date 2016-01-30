#!/usr/bin/env python -W ignore::DeprecationWarning
import pandas
from OneHotEncode import one_hot_dataframe
import matplotlib.pyplot as plt
import Functions
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

network_data = pandas.read_csv('D:/WINTER/EE239/Project1/network_backup_dataset.csv')

one_hot_data, _, _ = one_hot_dataframe(network_data, ['Day of Week', 'Work-Flow-ID','File Name'], replace=True)

one_hot_subset = one_hot_data[one_hot_data['Week #'] <= 3]
for num in range(0,4):
    Functions.plotWorkFlow(one_hot_subset,num, 'actual')

feature_cols = [col for col in one_hot_data.columns if col not in ['Size of Backup (GB)']]
X = one_hot_data[feature_cols]
y = one_hot_data['Size of Backup (GB)']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=3)

model = LinearRegression()
#Functions.callClassifier(model, X_train, y_train, X_test, y_test, feature_cols, 'Linear Regression')
model.fit(X_train, y_train)
print('Linear Regression - RMSE: %.4f' % (np.sqrt(np.sum((model.predict(X_test) - y_test) ** 2)/y_test.size)))

plt.scatter(model.predict(X_test), y_test,  color='green')
plt.xlabel('Fitted')
plt.ylabel('Actual')
plt.title('Fitted VS Actual')
#plt.show()
plt.savefig('NetBkpLRFitvsActual.png')
plt.clf()

plt.scatter(model.predict(X_test), (model.predict(X_test) - y_test),  color='blue')
#plt.hlines(y=0,xmin=0,xmax=50)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
#plt.show()
plt.savefig('NetBkpLRFitvsResidual.png')
plt.clf()

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
#
model.fit(X_train, y_train)
print('Random Forests after tuning - RMSE: %.4f' % (np.sqrt(np.sum((model.predict(X_test) - y_test) ** 2)/y_test.size)))

plt.scatter(model.predict(X_test), y_test,  color='green')
plt.xlabel('Fitted')
plt.ylabel('Actual')
plt.title('Fitted VS Actual RF')
#plt.show()
plt.savefig('RandomForestActualvsFit.png')
plt.clf()

pred = X_test
pred['Size of Backup (GB)'] = pandas.Series(model.predict(X_test), index=pred.index)
predicted_subset = pred[pred['Week #'] <= 3]
for num in range(0,4):
    Functions.plotWorkFlow(predicted_subset, num, 'predicted')
######################################################################################
#model = MLPRegressor()
#Functions.callClassifier(model, X_train, y_train, X_test, y_test, feature_cols, 'Neural Networks')
#####################################################################################

for num in range(0,4):
    data_workflow = data_workflow = one_hot_data[one_hot_data['Work-Flow-ID=work_flow_'+str(num)] == 1]
    X = data_workflow[feature_cols]
    y = data_workflow['Size of Backup (GB)']
    Functions.fitWorkFlow(LinearRegression(), X, y, num)

rmseList = []
for degree in [1, 2, 3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    rmse = np.sqrt(np.sum((model.predict(X_test) - y_test) ** 2)/y_test.size)
    rmseList.append(rmse)
    print('Polynomial Reg degree %d - RMSE: %.4f' % (degree, rmse))

plt.scatter(np.linspace(1,5,5), rmseList,  color='green')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('RMSE vs Degree Fixed')
#plt.show()
plt.savefig('RMSEvsDegreeFixed.png')
plt.clf()

rmseList = []
for degree in [1, 2, 3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    predicted = cross_val_predict(model, X, y, cv=10)
    rmse = np.sqrt(np.sum((predicted - y) ** 2)/y_test.size)
    rmseList.append(rmse)
    print('Polynomial Reg degree %d - RMSE: %.4f' % (degree, rmse))

plt.scatter(np.linspace(1,5,5), rmseList,  color='green')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('RMSE vs Degree CV')
#plt.show()
plt.savefig('RMSEvsDegreeCV.png')
plt.clf()