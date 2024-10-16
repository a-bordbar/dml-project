import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

# Load the MovieLens dataset
ratings_df = pd.read_csv("data/ratings.csv")

# Create a user-movie interaction matrix
num_users = ratings_df['userId'].nunique()
num_movies = ratings_df['movieId'].nunique()

# Create the user-movie interaction matrix
user_map = {user: i for i, user in enumerate(ratings_df['userId'].unique())}
movie_map = {movie: i for i, movie in enumerate(ratings_df['movieId'].unique())}

ratings_df['user_index'] = ratings_df['userId'].map(user_map)
ratings_df['movie_index'] = ratings_df['movieId'].map(movie_map)

interaction_matrix = coo_matrix((ratings_df['rating'], 
                                (ratings_df['user_index'], ratings_df['movie_index'])),
                                shape=(num_users, num_movies))



# Split into training and test sets
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Create training interaction matrix
train_matrix = coo_matrix((train_df['rating'], 
                           (train_df['user_index'], train_df['movie_index'])),
                          shape=(num_users, num_movies))



#Fill the missing values with the mean rating
mean_movie_rating = np.mean(train_matrix.toarray(), axis=0)
train_matrix_filled = train_matrix.toarray()
mean_movie_rating = np.true_divide(train_matrix_filled.sum(0), (train_matrix_filled != 0).sum(0))


for i in range(train_matrix_filled.shape[1]):  # Iterate over each movie (column)
    train_matrix_filled[train_matrix_filled[:, i] == 0, i] = mean_movie_rating[i]


# Apply SVD
svd = TruncatedSVD(n_components=10, random_state=42)
user_factors = svd.fit_transform(train_matrix)
movie_factors = svd.components_.T

# Reconstruct the ratings matrix
predicted_ratings = np.dot(user_factors, movie_factors.T)

# Evaluate on test set
# Ensure that all indices are within bounds and are integers
threshold = 3.0
test_df['binary_rating'] =(test_df['rating'] >= threshold).astype(int)  # 1 if rating >= 3, else 0

def debug_lambda(row):
    try:
        return predicted_ratings[int(row['user_index']), int(row['movie_index'])]
    except IndexError as e:
        print(f"IndexError: {e} at user_index: {row['user_index']}, movie_index: {row['movie_index']}")
        return np.nan  # Return NaN in case of error
test_df['predicted_rating'] = test_df.apply(debug_lambda, axis=1)

# Compute RMSE for collaborative filtering
auc = roc_auc_score(test_df['binary_rating'], test_df['predicted_rating'])
print(f"Collaborative Filtering AUC: {auc:.4f}")


rmse_cf = np.sqrt(((test_df['rating'] - test_df['predicted_rating']) ** 2).mean())
print(f"Collaborative Filtering RMSE: {rmse_cf:.4f}")
