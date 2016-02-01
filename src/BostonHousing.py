import pandas as pd
import Functions
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import cross_validation
import Plots

data = pd.read_csv('housing_data.csv')
feature_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X = data[feature_cols]
y = data.MEDV
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=3)
lm = LinearRegression()
predicted=Functions.callClassifierFeatures(lm, X_train, y_train, X_test, y_test, feature_cols, 'Linear Regression')
# Plotting
Plots.scatterPlot(predicted, y_test, 'Fitted', 'Actual', 'Fitted VS Actual LR', 'green', 'HousingLRScatterPlot')
Plots.residualPlot(predicted, (predicted - y_test), 'Fitted', 'Residual', 'Fitted VS Residual LR', 'blue', 'HousingLRResidualPlot')
# LR - cross val
predicted=Functions.callCrossVal(lm, X, y, 10, 'Linear Regression')
Plots.scatterPlot(predicted, y, 'Fitted', 'Actual', 'Fitted VS Actual LR-CV', 'green', 'HousingLRScatterPlotCV')
Plots.residualPlot(predicted, (predicted - y), 'Fitted', 'Residual', 'Fitted VS Residual LR-CV', 'blue', 'HousingLRResidualPlotCV')
# Polynomial Regression
Functions.polynomialRegression(lm, X_train, y_train, X_test, y_test, 6,'Linear Regression')
Functions.polynomialRegressionCV(lm, X, y, 10, 6, 'Linear Regression')
# Ridge
ridge = linear_model.RidgeCV(alphas=[0.1, 0.01, 0.001])
Functions.callClassifierFeatures(ridge, X_train, y_train, X_test, y_test,feature_cols, 'Ridge')
print("The tuned alpha value selected for Ridge is: %.4f" %ridge.alpha_)
Functions.callCrossVal(ridge, X, y, 10, 'Ridge')
# Lasso
lasso = linear_model.LassoCV(alphas=[0.1, 0.01, 0.001])
Functions.callClassifierFeatures(lasso, X_train, y_train, X_test, y_test,feature_cols, 'Lasso')
print("The tuned alpha value selected for Lasso is: %.4f" %lasso.alpha_)
Functions.callCrossVal(lasso, X, y, 10, 'Lasso')