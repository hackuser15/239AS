# download 100k dataset from http://grouplens.org/datasets/movielens/ and place in project directory
import os
from numpy import random

import pandas as pd
from Project3.Functions import *
import time
start_time = time.time()

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

ratings_path = "ml-100k/u.data"
abs_ratings_path = os.path.join(script_dir, ratings_path)
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(abs_ratings_path, sep='\t', names=r_cols)

#the movies file contains columns indicating the movie's genres
movies_path = "ml-100k/u.item"
abs_movies_path = os.path.join(script_dir, movies_path)
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
#m_cols = ['movie_id', 'title']
movies = pd.read_csv(abs_movies_path, sep='\t', names=m_cols, encoding = "ISO-8859-1")

matrix, weights = convertToMatrix(ratings)

############QUESTION1
# print('-------------------Q. 1--------------------')
# for k in [10, 50, 100]:
#     print('k: {}'. format(k))
#     U, V = nmfw(matrix, weights, k)

##########QUESTION2
# print('-------------------Q. 2 & 3--------------------')
# from sklearn.cross_validation import KFold
# kf = KFold(len(ratings), n_folds=10)
# scores = []
# precision = []
# recall = []
# loop_no = 1
# for train_index, test_index in kf:
#     print("CROSS VALIDATION : %s" %loop_no)
#     weights_new,_ = convertToMatrixKF(ratings,train_index)
#     prec_k = []
#     rec_k = []
#     # n_fav_movies = []
#     #call func
#     U, V = nmfw(matrix, weights_new, 100)
#     res_matrix = np.dot(U, V)
#     test_list = []
#     test_res_list = []
#     for index in test_index:
#         user = ratings.iloc[index]['user_id']-1
#         movie = ratings.iloc[index]['movie_id']-1
#         test_list.append(matrix[user, movie])
#         test_res_list.append(res_matrix[user, movie])
#     test_matrix = np.array(test_list)
#     test_res_matrix = np.array(test_res_list)
#     # print(test_matrix.shape)
#     # print(test_res_matrix.shape)
#     meanall = np.mean(np.absolute(np.subtract(res_matrix,matrix)))
#     mean = np.mean(np.absolute(np.subtract(test_res_matrix,test_matrix)))
#     # print("AVG ERROR IN FULL MATRIX %s:" %meanall)
#     print("AVG ERROR FOR TEST DATA IN THIS FOLD %s:" %mean)
#     scores.append(mean)
#
#     for k in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
#         predtest = np.where(test_matrix>k)
#         predtest = np.array(predtest)
#         predtestres = np.where(test_res_matrix>k)
#         predtestres = np.array(predtestres)
#         c = np.in1d(predtest,predtestres)
#
#         intersection = np.count_nonzero(c)
#         if predtest.size == 0:
#             prec,rec = 0.0,0.0
#         else:
#             prec = intersection/predtestres.size
#             rec = intersection/predtest.size
#         # print("Precision: %s" % prec)
#         # print("Recall: %s" % rec)
#         prec_k.append(prec)
#         rec_k.append(rec)
#         precision.append(prec)
#         recall.append(rec)
#         # n_fav_movies.append(intersection)
#         if(k==4):
#             print("Fold:%s Number of movies liked:%s" % (loop_no,intersection))
#     plotROC(rec_k,prec_k,'Recall','Precision','ROC_NoReg_Fold_'+str(loop_no))
#     print("Fold:%s Average Precision:%s" % (loop_no,np.mean(prec_k)))
#     print("Fold:%s Average Recall:%s" % (loop_no,np.mean(rec_k)))
#     loop_no = loop_no + 1
# print("MIN ERROR IN 10 FOLD %s:" %np.amin(scores))
# print("MAX ERROR IN 10 FOLD %s:" %np.amax(scores))
# print("AVG ERROR IN 10 FOLD %s:" %np.mean(scores))
# print("Average Precision over 10 folds:%s" % (np.mean(precision)))
# print("Average Recall over 10 folds:%s" % (np.mean(recall)))
# plotROC(recall,precision,'Recall','Precision','ROC_NoReg_Final')

#Q4
R_new, W_new = weights, matrix
# print('-------------------Q. 4 & 5--------------------')
# for k in [10, 50, 100]:
#     print('k: {}'. format(k))
#     U, V = nmfw(R_new, W_new, k)
#     R_hat = np.dot(U, V)

k = 100
n_iterations = 20
# for lambda_ in [0.01, 0.1, 1]:
# for lambda_ in [1]:
#     print('lambda: {}'. format(lambda_))
#     R_hat = weightedRegALS(R_new, lambda_, k, W_new, n_iterations)

lambda_ = 0.1
from sklearn.cross_validation import KFold
kf = KFold(len(ratings), n_folds=10)
scores = []
precision = []
recall = []
hit_rate_total = []
loop_no = 1
for train_index, test_index in kf:
    print("CROSS VALIDATION : %s" %loop_no)
    #print("TRAIN:", train_index, "TEST:", test_index)
    weights_new, matrix_new = convertToMatrixKF(ratings,train_index)
    _, matrix_test = convertToMatrixKF(ratings,test_index)
    prec_k = []
    rec_k = []

    res_matrix = weightedRegALS(weights_new, lambda_, k, matrix_new, n_iterations)
    # U, V = nmfw(weights_new, matrix_new,100)
    # res_matrix = np.dot(U,V)
    res_matrix = res_matrix*matrix
    res_matrix_test = res_matrix*matrix_test
    test_list = []
    test_res_list = []
    for index in test_index:
        user = ratings.iloc[index]['user_id']-1
        movie = ratings.iloc[index]['movie_id']-1
        test_list.append(matrix[user, movie])
        test_res_list.append(res_matrix[user, movie])
    test_matrix = np.array(test_list)
    test_res_matrix = np.array(test_res_list)
    meanall = np.mean(np.absolute(np.subtract(res_matrix,matrix)))
    mean = np.mean(np.absolute(np.subtract(test_res_matrix,test_matrix)))
    # print("AVG ERROR IN FULL MATRIX %s:" %meanall)
    print("AVG ERROR FOR TEST DATA IN THIS FOLD %s:" %mean)
    scores.append(mean)

    for k in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
        predtest = np.where(test_matrix>k)
        predtest = np.array(predtest)
        predtestres = np.where(test_res_matrix>k)
        predtestres = np.array(predtestres)
        c = np.in1d(predtest,predtestres)
        intersection = np.count_nonzero(c)
        if predtest.size == 0:
            prec,rec = 0.0,0.0
        else:
            prec = intersection/predtestres.size
            rec = intersection/predtest.size
        prec_k.append(prec)
        rec_k.append(rec)
        precision.append(prec)
        recall.append(rec)
    plotROC(rec_k,prec_k,'Recall','Precision','ROC_Regularized_Fold_'+str(loop_no))

    hit_rate = []
    L = 5.0
    for ind in range(0,res_matrix.shape[0]):
        movie_id = matrix_test[ind].argsort()[::-1][:L]
        movie_id_res = res_matrix_test[ind].argsort()[::-1][:L]
        if(np.count_nonzero(matrix_test[ind,movie_id]) == 0):
            continue
        c = np.in1d(movie_id,movie_id_res)
        intersection = np.count_nonzero(c)
        hit = intersection/L
        hit_rate.append(hit)
        hit_rate_total.append(hit)
    print("L=%s Fold:%s Average Precision:%s" % (L,loop_no,np.mean(hit_rate)))
    # if(loop_no==5):
        # break
    loop_no = loop_no + 1
print("MIN ERROR %s:" %np.amin(scores))
print("MAX ERROR %s:" %np.amax(scores))
print("AVG ERROR %s:" %np.mean(scores))
print("Average Precision all folds:%s" % (np.mean(precision)))
print("Average Recall over all folds:%s" % (np.mean(recall)))
print("Average Precision for top 5 recommendations over all folds:%s" % (np.mean(hit_rate_total)))
plotROC(recall,precision,'Recall','Precision','ROC_Regularized_Final')

# res_matrix = R_hat * matrix
# print("Recommended movie list for each user (Top 5)")
# for L in range(1,6):
#     hit_rate = []
#     false_rate = []
#     for ind in range(0,res_matrix.shape[0]):
#         movie_id = matrix[ind].argsort()[::-1][:L]
#         movie_id_res = res_matrix[ind].argsort()[::-1][:L]
#         if L == 5.0:
#             print("User ID: %s" %(ind+1))
#             print(movies.ix[movie_id_res+1]['title'])
#         c = np.in1d(movie_id,movie_id_res)
#         intersection = np.count_nonzero(c)
#         hit = intersection/L
#         false = 1-(intersection/L)
#         hit_rate.append(hit)
#         false_rate.append(false)
#     plotROC(false_rate,hit_rate,'False-Rate','Hit-Rate','False vs Hit for L='+str(L))
#
# print("--- %s seconds ---" % (time.time() - start_time))