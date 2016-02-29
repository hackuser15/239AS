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
#print(matrix)
#print(weights)
#print(matrix.shape)
#print(weights.shape)
nz = np.where(matrix>0)
row = nz[0]
col = nz[1]
print(row)
print(col)
c = list(zip(row, col))
random.shuffle(c)
a, b = zip(*c)
#
print(a)
print(b)
print(len(a))
# print(row.shape)
# print(col.shape)
#print(matrix[np.where(matrix>0)])
#print(matrix[np.where(matrix>0)].shape)
from sklearn.cross_validation import KFold
#kf = KFold(nz[0].shape[0], n_folds=10)
kf = KFold(len(a), n_folds=10)
print(len(kf))
scores = []
for train_index, test_index in kf:
    #matrix1 = matrix.copy()
    weights1 = weights.copy()
    #print("TRAIN:", train_index, "TEST:", test_index)
    # print(row[test_index])
    # print(col[test_index])
    # print("Test")
    # print(matrix[row[test_index],col[test_index]])
    # print(weights1[row[test_index],col[test_index]])
    # #matrix[row[test_index],col[test_index]] = 0

    weights1[row[test_index],col[test_index]] = 0
    #matrix1[row[test_index],col[test_index]] = 0

    # print(matrix[row[test_index],col[test_index]])
    # print(weights1[row[test_index],col[test_index]])
    # print("Train")
    # print(matrix[row[train_index],col[train_index]])
    # print(weights1[row[train_index],col[train_index]])
    # print(matrix[row[train_index],col[train_index]])
    # print(weights1[row[train_index],col[train_index]])
    #for index in test_index:
        #weights[index] = 0

    U, V = nmfw(matrix, weights1, 100)
    res_matrix = np.dot(U, V)
    test_matrix = matrix[row[test_index],col[test_index]]
    test_res_matrix = res_matrix[row[test_index],col[test_index]]
    # print(weights1[row[test_index],col[test_index]])
    # print(test_matrix)
    # print(test_res_matrix)
    train_matrix = matrix[row[train_index],col[train_index]]
    train_matrix_res = res_matrix[row[train_index],col[train_index]]
    # print(weights1[row[train_index],col[train_index]])
    # print(train_matrix)
    # print(train_matrix_res)
    sum = np.sum(np.absolute(np.subtract(test_res_matrix,test_matrix)))
    meanall = np.mean(np.absolute(np.subtract(res_matrix,matrix)))
    meantr = np.mean(np.absolute(np.subtract(train_matrix_res,train_matrix)))
    mean = np.mean(np.absolute(np.subtract(test_res_matrix,test_matrix)))
    print(mean)
    print(meanall)
    print(meantr)
    scores.append(mean)
    #weights = weights1
    #matrix = matrix1
print(np.amin(scores))
print(np.mean(scores))

# precision and recall
print(test_matrix)
print(test_res_matrix)
precision = []
recall = []
for k in [1, 2, 3, 4]:
    print(k)
    print(test_matrix.shape)
    print(test_res_matrix.shape)
    predtest = np.where(test_matrix>k)
    predtest = np.array(predtest)
    print(predtest)
    print(predtest.shape)
    predtestres = np.where(test_res_matrix>k)
    predtestres = np.array(predtestres)
    print(predtestres.shape)
    print(predtestres)
    c = np.in1d(predtest,predtestres)
    print(c)
    intersection = np.count_nonzero(c)
    prec = intersection/predtestres.shape[1]
    rec = intersection/predtest.shape[1]
    print("Precision: %s" % prec)
    print("Recall: %s" % rec)
    precision.append(prec)
    recall.append(rec)
    print()
print(precision)
print(recall)
plotROC(precision,recall,'ROC')
#
# array([ True,  True,  True], dtype=bool)

# for k in [10, 50, 100]:
#     print('k: {}'. format(k))
#     U, V = nmfw(matrix, weights, k)
#     #U, V = nmf(matrix, k)
#
# matrix1 = np.dot(U, V)
# print(matrix1)
# print(matrix1.shape)
# model = NMF(n_components=10, init='random', random_state=0)
# model.fit(matrix)
