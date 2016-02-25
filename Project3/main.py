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

# the movies file contains columns indicating the movie's genres
# movies_path = "ml-100k/u.item"
# abs_movies_path = os.path.join(script_dir, movies_path)
# m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
# movies = pd.read_csv(abs_movies_path, sep='|', names=m_cols, encoding = "ISO-8859-1")

# create one merged DataFrame
# movie_ratings = pd.merge(movies, ratings)
# lens = pd.merge(movie_ratings, users)

matrix, weights = convertToMatrix(ratings)
for k in [10, 50, 100]:
    print('k: {}'. format(k))
    U, V = nmf(matrix, k)

# model = NMF(n_components=10, init='random', random_state=0)
# model.fit(matrix)
