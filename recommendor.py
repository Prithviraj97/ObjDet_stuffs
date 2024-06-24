import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import seaborn as sns

class RecommenderSystem:
    def __init__(self, ratings):
        self.ratings = ratings
        self.user_ratings_mean = None
        self.predictions = None

    def preprocess_data(self):
        R = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        R_matrix = R.values
        self.user_ratings_mean = np.mean(R_matrix, axis=1)
        R_demeaned = R_matrix - self.user_ratings_mean.reshape(-1, 1)
        return R_demeaned, R

    def compute_svd(self, R_demeaned):
        U, sigma, Vt = svds(R_demeaned, k=50)
        sigma = np.diag(sigma)
        return U, sigma, Vt

    def make_predictions(self, U, sigma, Vt):
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean.reshape(-1, 1)
        self.predictions = pd.DataFrame(all_user_predicted_ratings, columns=self.ratings['movieId'].unique())
    
    def recommend_movies(self, user_id, num_recommendations):
        user_row_number = user_id - 1
        sorted_user_predictions = self.predictions.iloc[user_row_number].sort_values(ascending=False)
        user_data = self.ratings[self.ratings.userId == user_id]
        recommendations = self.ratings[~self.ratings['movieId'].isin(user_data['movieId'])]
        recommendations = recommendations.merge(pd.DataFrame(sorted_user_predictions).reset_index(), on='movieId')
        recommendations = recommendations.rename(columns={user_row_number: 'Predictions'}).sort_values('Predictions', ascending=False)
        recommendations = recommendations.merge(movies, on='movieId')  # Merge with movie titles
        return recommendations[['movieId', 'title', 'genre', 'Predictions']].head(num_recommendations)

    def evaluate_model(self, R, R_demeaned):
        U, sigma, Vt = self.compute_svd(R_demeaned)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean.reshape(-1, 1)
        rmse = np.sqrt(mean_squared_error(R.values, all_user_predicted_ratings))
        mae = mean_absolute_error(R.values, all_user_predicted_ratings)
        return rmse, mae

    def visualize_data(self, R, R_demeaned, U, sigma, Vt):
        plt.figure(figsize=(10,6))
        sns.heatmap(R.values, cmap='coolwarm')
        plt.title('User-Item Interaction Matrix')
        plt.xlabel('Item')
        plt.ylabel('User')
        plt.show()

        precision, recall, _ = precision_recall_curve(R.values.flatten(), np.dot(np.dot(U, sigma), Vt).flatten())
        auc_precision_recall = auc(recall, precision)
        plt.figure(figsize=(10,6))
        plt.plot(recall, precision, color='blue', label=f'AUC: {auc_precision_recall:.2f}')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

def user_based_collaborative_filtering(ratings, user_id, num_recommendations):
    user_data = ratings[ratings.userId == user_id]
    similar_users = ratings[ratings.userId != user_id]
    similar_users['similarity'] = similar_users.apply(lambda row: len(set(row['movieId']) & set(user_data['movieId'])), axis=1)
    similar_users = similar_users.sort_values('similarity', ascending=False)
    recommendations = similar_users.merge(movies, on='movieId')
    return recommendations[['movieId', 'title', 'genre']].head(num_recommendations)

def main():
    global movies
    ratings_dict = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'movieId': [1, 2, 3, 1, 2, 4, 2, 3, 4],
        'rating': [5, 4, 3, 4, 5, 2, 2, 4, 5]
    }
    ratings = pd.DataFrame(ratings_dict)
    movies_dict = {
        'movieId': [1, 2, 3, 4],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
        'genre': ['Action', 'Comedy', 'Drama', 'Action']
    }
    movies = pd.DataFrame(movies_dict)
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    rs = RecommenderSystem(train_ratings)
    R_demeaned, R = rs.preprocess_data()
    U, sigma, Vt = rs.compute_svd(R_demeaned)
    rs.make_predictions(U, sigma, Vt)
    rmse, mae = rs.evaluate_model(R, R_demeaned)
    print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}')
    rs.visualize_data(R, R_demeaned, U, sigma, Vt)
    user_id = 1
    num_recommendations = 5
    recommendations = rs.recommend_movies(user_id, num_recommendations)
    print(recommendations)
    user_based_recommendations = user_based_collaborative_filtering(train_ratings, user_id, num_recommendations)
    print(user_based_recommendations)

if __name__ == "__main__":
    main()