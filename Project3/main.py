# download 100k dataset from http://grouplens.org/datasets/movielens/ and place in project directory
import os
from numpy import random

import pandas as pd
from Project3.Functions import *
from sklearn.decomposition import NMF

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
# user_path = "ml-100k/u.user"
# abs_user_path = os.path.join(script_dir, user_path)
# pass in column names for each CSV
# u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
# users = pd.read_csv(abs_user_path, sep='|', names=u_cols)

ratings_path = "ml-100k/u.data"
abs_ratings_path = os.path.join(script_dir, ratings_path)
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(abs_ratings_path, sep='\t', names=r_cols)
#print(ratings)

# the movies file contains columns indicating the movie's genres
# movies_path = "ml-100k/u.item"
# abs_movies_path = os.path.join(script_dir, movies_path)
# m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
# movies = pd.read_csv(abs_movies_path, sep='|', names=m_cols, encoding = "ISO-8859-1")

# create one merged DataFrame
# movie_ratings = pd.merge(movies, ratings)
# lens = pd.merge(movie_ratings, users)

matrix, weights = convertToMatrix(ratings)

############QUESTION1
print('-------------------Q. 1--------------------')
for k in [10, 50, 100]:
    print('k: {}'. format(k))
    U, V = nmfw(matrix, weights, k)
    matrix1 = np.dot(U, V)
    print(np.mean(np.absolute(np.subtract(matrix1,matrix))))
    U, V = nmf(matrix, k)


# model = NMF(n_components=10, init='random', random_state=0)
# model.fit(matrix)

###########QUESTION2
print('-------------------Q. 2 & 3--------------------')
from sklearn.cross_validation import KFold
kf = KFold(len(ratings), n_folds=10)
print(len(kf))
scores = []
precision = []
recall = []
loop_no = 1
for train_index, test_index in kf:
    print("CROSS VALIDATION : %s" %loop_no)

    #print("TRAIN:", train_index, "TEST:", test_index)
    weights = convertToMatrixKF(ratings,train_index)
    prec_k = []
    rec_k = []
    #call func
    U, V = nmfw(matrix, weights, 100)
    res_matrix = np.dot(U, V)
    test_list = []
    test_res_list = []
    for index in test_index:
        user = ratings.iloc[index]['user_id']-1
        movie = ratings.iloc[index]['movie_id']-1
        test_list.append(matrix[user, movie])
        test_res_list.append(res_matrix[user, movie])
    test_matrix = np.array(test_list)
    test_res_matrix = np.array(test_res_list)
    # print(test_matrix.shape)
    # print(test_res_matrix.shape)
    meanall = np.mean(np.absolute(np.subtract(res_matrix,matrix)))
    mean = np.mean(np.absolute(np.subtract(test_res_matrix,test_matrix)))
    print("AVG ERROR IN FULL MATRIX %s:" %meanall)
    print("AVG ERROR FOR TEST DATA IN THIS FOLD %s:" %mean)
    scores.append(mean)

    for k in [3, 3.5, 4]:
        predtest = np.where(test_matrix>k)
        predtest = np.array(predtest)
        predtestres = np.where(test_res_matrix>k)
        predtestres = np.array(predtestres)
        c = np.in1d(predtest,predtestres)
        #print(c)
        intersection = np.count_nonzero(c)
        #print(predtest.size)
        #print(predtestres.size)
        if predtest.size == 0:
            prec,rec = 0.0,0.0
        else:
            prec = intersection/predtestres.size
            rec = intersection/predtest.size
        print("Precision: %s" % prec)
        print("Recall: %s" % rec)
        prec_k.append(prec)
        rec_k.append(rec)
        precision.append(prec)
        recall.append(rec)
    plotROC(prec_k,rec_k,'Precision','Recall','ROC_NoReg_Fold_'+str(loop_no))
    loop_no = loop_no + 1
print("MIN ERROR IN 10 FOLD %s:" %np.amin(scores))
print("MAX ERROR IN 10 FOLD %s:" %np.amax(scores))
print("AVG ERROR IN 10 FOLD %s:" %np.mean(scores))
print(precision)
print(recall)
plotROC(precision,recall,'Precision','Recall','ROC_NoReg_Final')
# print(len(precision))
# print(len(recall))

#Q4
R_new, W_new = weights, matrix
print('-------------------Q. 4--------------------')
for k in [10, 50, 100]:
    print('k: {}'. format(k))
    U, V = nmfw(R_new, W_new, k)
    R_hat = np.dot(U, V)

k = 100
n_iterations = 20
for lambda_ in [0.01, 0.1, 1]:
    print('lambda: {}'. format(lambda_))
    R_hat = weightedRegALS(R_new, lambda_, k, W_new, n_iterations)

lambda_ = 0.1
from sklearn.cross_validation import KFold
kf = KFold(len(ratings), n_folds=10)
# print(len(kf))
scores = []
precision = []
recall = []
loop_no = 1
for train_index, test_index in kf:
    print("CROSS VALIDATION : %s" %loop_no)
    #print("TRAIN:", train_index, "TEST:", test_index)
    weights = convertToMatrixKF(ratings,train_index)
    matrix_new = matrix[weights > 0.0]
    prec_k = []
    rec_k = []
    #call func

    U, V = weightedRegALS(weights, lambda_, k, matrix_new, n_iterations)
    res_matrix = np.dot(U, V)
    test_list = []
    test_res_list = []
    for index in test_index:
        user = ratings.iloc[index]['user_id']-1
        movie = ratings.iloc[index]['movie_id']-1
        test_list.append(matrix[user, movie])
        test_res_list.append(res_matrix[user, movie])
    test_matrix = np.array(test_list)
    test_res_matrix = np.array(test_res_list)
    # print(test_matrix.shape)
    # print(test_res_matrix.shape)
    meanall = np.mean(np.absolute(np.subtract(res_matrix,matrix)))
    mean = np.mean(np.absolute(np.subtract(test_res_matrix,test_matrix)))
    print("AVG ERROR IN FULL MATRIX %s:" %meanall)
    print("AVG ERROR FOR TEST DATA IN THIS FOLD %s:" %mean)
    scores.append(mean)

    for k in [3, 3.5, 4]:
        predtest = np.where(test_matrix>k)
        predtest = np.array(predtest)
        predtestres = np.where(test_res_matrix>k)
        predtestres = np.array(predtestres)
        c = np.in1d(predtest,predtestres)
        #print(c)
        intersection = np.count_nonzero(c)
        #print(predtest.size)
        #print(predtestres.size)
        if predtest.size == 0:
            prec,rec = 0.0,0.0
        else:
            prec = intersection/predtestres.size
            rec = intersection/predtest.size
        print("Precision: %s" % prec)
        print("Recall: %s" % rec)
        prec_k.append(prec)
        rec_k.append(rec)
        precision.append(prec)
        recall.append(rec)
    plotROC(prec_k,rec_k,'Precision','Recall','ROC_Regularized_Fold_'+str(loop_no))
    loop_no = loop_no + 1
print("MIN ERROR IN 10 FOLD %s:" %np.amin(scores))
print("MAX ERROR IN 10 FOLD %s:" %np.amax(scores))
print("AVG ERROR IN 10 FOLD %s:" %np.mean(scores))
print(precision)
print(recall)
plotROC(precision,recall,'Precision','Recall','ROC_Regularized_Final')
