import numpy as np
from scipy import linalg
from numpy import dot
import matplotlib.pyplot as plt

def convertToMatrix(ratings):
    nr = np.max(ratings['user_id'])
    nc = np.max(ratings['movie_id'])
    matrix = np.zeros((nr, nc))
    weights = np.zeros((nr, nc))
    for index, row in ratings.iterrows():
        user = row['user_id']-1
        movie = row['movie_id']-1
        rating = row['rating']
        matrix[user, movie] = rating
        weights[user, movie] = 1
    return (matrix, weights)

def convertToMatrixKF(ratings,train_index):
    nr = np.max(ratings['user_id'])
    nc = np.max(ratings['movie_id'])
    matrix = np.zeros((nr, nc))
    weights = np.zeros((nr, nc))
    for index in train_index:
        user = ratings.iloc[index]['user_id']-1
        movie = ratings.iloc[index]['movie_id']-1
        rating = ratings.iloc[index]['rating']
        matrix[user, movie] = rating
        weights[user, movie] = 1
    return weights, matrix

def nmfw(X, weights, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    eps = 1e-5

    mask = weights

    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    bool_mask = mask.astype(bool)

    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)

        if i == max_iter:
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            abs = np.mean(mask*np.absolute(np.subtract(X,X_est)))
            print('Total Squared error', np.round(curRes**2, 4))
            print('Mean Absolute error', np.round(abs, 4))
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y

def plotROC(X,Y,xlabel,ylabel,name):
    plt.figure()
    plt.plot(X, Y, label=name, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Plot %s' %name)
    #plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(name+'.png')
    plt.clf()

def weightedRegALS(Q, lambda_, n_factors, W, n_iterations):
    m, n = Q.shape

    X = 5 * np.random.rand(m, n_factors)
    Y = np.linalg.lstsq(X, Q)[0]

    for ii in range(n_iterations):
        for u, Wu in enumerate(W):
            X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                                   np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
        for i, Wi in enumerate(W.T):
            Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                     np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
        if(ii == n_iterations - 1):
            print('Total Squared Error {}'.format(get_error(np.dot(X, Y), Q, W)))
            print('Mean Absolute Error {}'.format(get_abs_error(np.dot(X, Y), Q, W)))
    weighted_Q_hat = np.dot(X,Y)
    return weighted_Q_hat

def get_error(R_hat, R, W):
    return np.sum((W * (R_hat - R))**2)

def get_abs_error(R_hat, R, W):
    tmp = W *np.abs(R_hat - R)
    return np.mean(tmp[W > 0.0])