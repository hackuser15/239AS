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
        #matrix[user, movie] = rating
        weights[user, movie] = 1
    return weights


def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    # X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    bool_mask = mask.astype(bool)
    for i in range(columns):
        Y[:,i] = linalg.lstsq(A[bool_mask[:,i],:], X[bool_mask[:,i],i])[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i == max_iter:
            # print('Iteration {}:'.format(i))
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print('fit residual', np.round(fit_residual, 4))
            print('total residual', np.round(curRes, 4))
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y

def nmfw(X, weights, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-5

    print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    # X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = weights
    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    bool_mask = mask.astype(bool)

    # for i in range(columns):
    #     Y[:,i] = linalg.lstsq(A[bool_mask[:,i],:], X[bool_mask[:,i],i])[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i == max_iter:
            # print('Iteration {}:'.format(i))
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print('fit residual', np.round(fit_residual, 4))
            print('total residual', np.round(curRes, 4))
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y

def plotROC(fpr,tpr,name):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Receiver operating characteristic for %s' %name)
    #plt.legend(loc="lower right")
    plt.show()
    #plt.savefig(name+'.png')
    #plt.clf()

def weightedRegALS(Q, lambda_, n_factors, W, n_iterations):
    m, n = Q.shape

    X = 5 * np.random.rand(m, n_factors)
    Y = np.linalg.lstsq(X, Q)[0]

    weighted_errors = []
    totalError =0
    for ii in range(n_iterations):
        for u, Wu in enumerate(W):
            X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                                   np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
        for i, Wi in enumerate(W.T):
            Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                     np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
        # weighted_errors.append(get_error(Q, X, Y, W))
        # totalError += get_error(Q, X, Y, W)
        if(ii == n_iterations - 1):
            print('Total Error {}'.format(totalError))
    weighted_Q_hat = np.dot(X,Y)
    return weighted_Q_hat