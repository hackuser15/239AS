import pandas
from OneHotEncode import one_hot_dataframe
import matplotlib.pyplot as plt
import Functions
import Plots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.grid_search import RandomizedSearchCV,GridSearchCV
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

######################################################################################

network_data = pandas.read_csv('network_backup_dataset.csv')

one_hot_data, _, _ = one_hot_dataframe(network_data, ['Day of Week', 'Work-Flow-ID','File Name'], replace=True)

one_hot_subset = one_hot_data[one_hot_data['Week #'] <= 3]

for num in range(0,5):
    Plots.plotWorkFlow(one_hot_subset,num, 'actual')

feature_cols = [col for col in one_hot_data.columns if col not in ['Size of Backup (GB)']]
X = one_hot_data[feature_cols]
y = one_hot_data['Size of Backup (GB)']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=3)

model = LinearRegression()
Functions.callClassifierFeatures(model, X_train, y_train, X_test, y_test, feature_cols, 'Linear Regression')

pred = model.predict(X_test)
Plots.scatterPlot(pred, y_test, 'Fitted','Actual','Fitted VS Actual','green','NetBkpLRFitvsActual')

Plots.residualPlot(pred, pred - y_test, 'Fitted','Resuduals','Fitted VS Residual','green','NetBkpLRFitvsResidual')

Functions.callCrossVal(model, X, y, 10, 'Linear Regression')
######################################################################################
model = RandomForestRegressor(n_estimators=20, max_depth=4, max_features='auto')
Functions.callClassifier(model, X_train, y_train, X_test, y_test,'Random Forests Initial')
model.fit(X_train, y_train)
print('Random Forests Initial - RMSE: %.4f' % (np.sqrt(np.sum((model.predict(X_test) - y_test) ** 2)/y_test.size)))

clf = RandomForestRegressor()
param_dist = {"n_estimators":sp_randint(1, 100),
              "max_depth": sp_randint(1, 10),
              "max_features": sp_randint(1, 45),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False]}

#run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(X, y)
print('Best Parameters for Random forest:')
print(random_search.best_params_)

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

# pred_rf = Functions.callClassifier(random_search, X_train, y_train, X_test, y_test,'Random Forests after tuning')

model = RandomForestRegressor(n_estimators=50, min_samples_split= 4, max_features= 41, min_samples_leaf= 4, bootstrap= True, max_depth= 9)
pred_rf = Functions.callClassifier(model, X_train, y_train, X_test, y_test,'Random Forests after tuning')
model.fit(X_train, y_train)
print('Random Forests after tuning - RMSE: %.4f' % (np.sqrt(np.sum((model.predict(X_test) - y_test) ** 2)/y_test.size)))

Plots.scatterPlot(pred_rf, y_test, 'Fitted','Actual','Fitted VS Actual RF','green','RandomForestActualvsFit')

Plots.residualPlot(pred_rf, pred_rf - y_test, 'Fitted','Resuduals','Fitted VS Residual','green','RandomForestFitvsResidual')

pred = X_test.copy()
pred['Size of Backup (GB)'] = pandas.Series(pred_rf, index=pred.index)
predicted_subset = pred[pred['Week #'] <= 3]
for num in range(0,5):
    Plots.plotWorkFlow(predicted_subset, num, 'predicted')
#######################################################################################
ds = SupervisedDataSet(45, 1)
ds.setField( 'input', X_train )
y_train_nn = y_train.copy().reshape(-1, 1)
ds.setField( 'target', y_train_nn )

ds_test = SupervisedDataSet(45, 1)
ds_test.setField( 'input', X_test)
y_test_nn = y_test.copy().reshape( -1, 1 )
ds_test.setField( 'target', y_test_nn )

for hidden in range(1,5):
    for epoch in range(10,40,10):
        hidden_size = hidden   # arbitrarily chosen

        net = buildNetwork(45, hidden_size, 1, bias = True)
        trainer = BackpropTrainer( net, ds )

        #trainer.trainOnDataset( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )
        # trainer.trainOnDataset(ds, 2)
        trainer.trainUntilConvergence(maxEpochs = epoch)

        p = net.activateOnDataset( ds_test )
        print('Neural Network - Hidden size: %d Epchs: %d RMSE: %.4f' % (hidden, epoch, np.sqrt(np.sum((p - y_test_nn) ** 2)/y_test.size)))
# #####################################################################################

for num in range(0,5):
    data_workflow =  one_hot_data[one_hot_data['Work-Flow-ID=work_flow_'+str(num)] == 1]
    X = data_workflow[feature_cols]
    y = data_workflow['Size of Backup (GB)']
    Functions.fitWorkFlow(LinearRegression(), X, y, num)

Functions.polynomialRegression(LinearRegression(), X_train, y_train, X_test, y_test, 3,'NetworkBackupPoly')

Functions.polynomialRegressionCV(LinearRegression(), X, y, 10, 3, 'NetworkBackupPolyCV')
