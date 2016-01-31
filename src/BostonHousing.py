import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Functions
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import cross_validation
import Plots

data = pd.read_csv('housing_data.csv')
feature_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X = data[feature_cols]
y = data.MEDV
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=3)
lm = LinearRegression()
Functions.callClassifier(lm, X_train, y_train, X_test, y_test, feature_cols, 'Linear Regression')

# Plotting
#Plots.scatterPlot(lm.predict(X_test), y_test, 'abc', 'def', 'title', 'green')
plt.scatter(lm.predict(X_test), y_test,  color='green')
#plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',linewidth=3)
plt.xlabel('Fitted')
plt.ylabel('Actual')
plt.title('Fitted VS Actual')
plt.show()
plt.scatter(lm.predict(X_test), (lm.predict(X_test) - y_test),  color='blue')
plt.hlines(y=0,xmin=0,xmax=50)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
plt.show()
# LR - cross val
Functions.callCrossVal(lm, X, y, 10, 'Linear Regression')

# Polynomial Regression
 #   model = make_pipeline(PolynomialFeatures(degree), lm)
Functions.polynomialRegression(lm, X_train, y_train, X_test, y_test, 'Poly Linear Regression')
Functions.polynomialRegressionCV(lm, X, y, 10, 'Linear Regression')
# Ridge
ridge = linear_model.RidgeCV(alphas=[0.1, 0.01, 0.001])
Functions.callClassifier(ridge, X_train, y_train, X_test, y_test, feature_cols, 'Ridge')
# Ridge predicted results with CV
Functions.callCrossVal(ridge, X, y, 10, 'Ridge')

# Lasso
lasso = linear_model.LassoCV(alphas=[0.1, 0.01, 0.001])
Functions.callClassifier(lasso, X_train, y_train, X_test, y_test, feature_cols, 'Lasso')
# Lasso predicted results with CV
Functions.callCrossVal(lasso, X, y, 10, 'Lasso')