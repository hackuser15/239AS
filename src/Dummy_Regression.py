import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, LinearRegression, Lars, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

class Regression:
    def linearRegression(self, train_data_features,train_review_rating, data):
        # Create linear regression object
        regr = LinearRegression()

        # Train the model using the training sets
        regr.fit(train_data_features, train_review_rating)

        return regr

    def svmRegressionRbf(self, train_data_features,train_review_rating, data):
        C = 1
        gamma = 0.1
        if data == 'bow':
            C =np.exp(1)
            gamma = 0.1
        elif data == 'usrbiz':
            C = np.exp(3)
            gamma= np.exp(3)
        elif data == 'merged':
            C = np.exp(3)
            gamma = np.exp(-1)

        svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma,cache_size=7000)
        #svr_rbf = GridSearchCV(SVR(kernel='rbf'), param_grid={"C": np.logspace(-3, 3, num=7, base= np.exp(1)),"gamma": np.logspace(-2, 2, 5)})
        svr_rbf.fit(train_data_features, train_review_rating)
        return svr_rbf

    def ridgeRegression(self, train_data_features,train_review_rating, data):
        alpha = 0.1
        normalize = False
        if data == 'bow':
            alpha = np.exp(1)
            normalize = False
        elif data == 'usrbiz':
            alpha = 1.0
            normalize = True
        elif data == 'merged':
            alpha = np.exp(2)
            normalize = True
        rr = Ridge(alpha=alpha, normalize= normalize)
        #rr = GridSearchCV(Ridge(), param_grid={"alpha": np.logspace(-7, 3, num=11, base = np.exp(1))})
        rr.fit(train_data_features, train_review_rating)
        #rss = np.sum((predicted_rating_krr - test_review_rating) ** 2)
        return rr


    def kernelRidgeRegressionPoly(self, train_data_features,train_review_rating, data):
        degree = 3
        alpha = 1.0
        if data == 'bow':
            degree = 4
            alpha = np.exp(-4)
        elif data == 'usrbiz':
            degree = 2
            alpha = np.exp(3)
        elif data == 'merged':
            degree = 2
            alpha = np.exp(2)
        #alphas = np.logspace(-4, 4, num=8)
        #degrees = np.array([2,3,4])
        #krr_poly = GridSearchCV(KernelRidge(kernel = 'poly'), param_grid=dict(alpha=alphas,degree=degrees))
        krr_poly = KernelRidge(alpha=alpha, degree=degree, kernel='poly')
        krr_poly.fit(train_data_features, train_review_rating)
        return krr_poly

    def bayesianRegression(self, train_data_features,train_review_rating, data):
        br = BayesianRidge(compute_score=True, lambda_1=1.e-2,lambda_2=1.e-2)
        br.fit(train_data_features, train_review_rating)
        return br

    def nearestNeighbour(self, train_data_features,train_review_rating, data):
        nn = 5
        if data == 'bow':
            nn =91
        elif data == 'usrbiz':
            nn = 351
        elif data == 'merged':
            nn = 351
        #nn= np.linspace(349, 353, num=5)
        #knn = GridSearchCV(KNeighborsRegressor(weights='uniform'), param_grid=dict(n_neighbors =nn))
        knn = KNeighborsRegressor(n_neighbors =nn, weights='uniform')
        #knn = GridSearchCV(KNeighborsRegressor(weights='uniform'), param_grid=dict(n_neighbors =nn))
        knn.fit(train_data_features, train_review_rating)
        return knn
        #print(knn.best_params_)

    def performance(self, predicted_label, actual_label):
        n = np.size(predicted_label)
        a_mean = [np.mean(actual_label) for x in range(len(actual_label))]
        rss = np.sum((predicted_label - actual_label) ** 2)
        rms = np.sqrt(np.sum((predicted_label - actual_label) ** 2)/n)
        rae = (np.sum(np.absolute(predicted_label - actual_label)))/(np.sum(np.absolute(a_mean - actual_label)))
        rse = (np.sum(np.square(predicted_label - actual_label)))/(np.sum(np.square(a_mean - actual_label)))
        mae = np.mean(np.absolute(predicted_label - actual_label))
        roundedRating = [ round(elem, 0) for elem in predicted_label ]
        nmatch = 0
        for i in range(0,len(predicted_label)):
          if roundedRating[i] == actual_label[i]:
              nmatch += 1
        print("Residual sum of squares: %.2f" % (rss))
        print("Root Mean Squared Error: %.2f" % (rms))
        print("Relative Absolute Error: %.2f" % (rae))
        print("Root Relative Squared Error: %.2f" % (rse))
        print("Mean Absolue Error: %.2f" % (mae))
        print("Accuracy : %.2f" % (nmatch*100/(len(predicted_label))))