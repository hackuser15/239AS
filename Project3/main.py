# download 100k dataset from http://grouplens.org/datasets/movielens/ and place in project directory
import os
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
print(row.shape)
print(col.shape)
#print(matrix[np.where(matrix>0)])
#print(matrix[np.where(matrix>0)].shape)
from sklearn.cross_validation import KFold
kf = KFold(nz[0].shape[0], n_folds=10)
print(len(kf))
scores = []
for train_index, test_index in kf:
    matrix1 = matrix
    weights1 = weights
    #print("TRAIN:", train_index, "TEST:", test_index)
    print(train_index.shape)
    print(test_index.shape)
    print(row[test_index])
    print(col[test_index])
    print(matrix[row[test_index],col[test_index]])
    print(weights[row[test_index],col[test_index]])
    matrix[row[test_index],col[test_index]] = 0
    #weights[row[test_index],col[test_index]] = 0
    print(matrix[row[test_index],col[test_index]])
    print(weights[row[test_index],col[test_index]])
    print(weights.shape)
    #for index in test_index:
        #weights[index] = 0
    U, V = nmfw(matrix, weights, 100)
    res_matrix = np.dot(U, V)
    test_matrix = matrix[row[test_index],col[test_index]]
    test_res_matrix = res_matrix[row[test_index],col[test_index]]
    print(test_matrix)
    print(test_res_matrix)
    print(test_matrix.shape)
    print(test_res_matrix.shape)
    sum = np.sum(np.abs(np.subtract(test_res_matrix,test_matrix)))
    print(sum)
    scores.append(sum)
    weights = weights1
    matrix = matrix1
    #y_train, y_test = y[train_index], y[test_index]

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
