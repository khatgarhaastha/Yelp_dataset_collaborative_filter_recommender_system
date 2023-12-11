# Collaborative Filtering and Neural Network-Based Recommender System

## Project Overview

This project implements a recommender system using two distinct approaches: Collaborative Filtering and Artificial Neural Networks (ANN). The system is designed to recommend restaurants based on user reviews from the Yelp dataset. The objective is to compare the performance of traditional collaborative filtering techniques against a more complex neural network approach in terms of accuracy and efficiency.

## Data

The Yelp dataset, comprising business and review data, is used. The business dataset includes information such as business IDs, names, and categories. The review dataset contains user IDs, business IDs, stars, and review dates.

## Methods

### 1. Collaborative Filtering
Collaborative filtering is implemented using cosine similarity to predict user preferences based on ratings given by similar users. The key steps include:
- Reading and chunking data from JSON files.
- Filtering relevant businesses and reviews.
- Merging datasets and creating a user-item matrix.
- Computing the cosine similarity matrix.
- Predicting ratings and evaluating the model's performance using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

### 2. Neural Network-Based Approach
The ANN approach involves building a model using Keras with TensorFlow backend. The model includes:
- Embedding layers for users and restaurants.
- Dot product of user and restaurant embeddings to capture interactions.
- Regularization to prevent overfitting.
- Sigmoid activation to scale the output.
- Adam optimizer for training the model.

Data is encoded using `LabelEncoder` before being fed into the model. The performance is again evaluated using RMSE and MAE.

## Requirements

- Python 3.11
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow
- Keras

## Usage

To run the project, execute the `main()` function. The function iteratively reads the data in chunks, applies both the collaborative filtering and neural network models, and prints the performance metrics.

## Comparison of Approaches

The project provides an insightful comparison between collaborative filtering and neural networks in the context of a recommender system. It illustrates the trade-offs between complexity, interpretability, and accuracy.

## Potential Improvements

- Experimenting with different architectures and hyperparameters for the neural network.
- Incorporating more features into the model, such as text reviews or temporal data.
- Implementing a more advanced collaborative filtering technique, such as matrix factorization.

## Acknowledgements

This project uses the Yelp dataset for educational purposes. The methods and approaches are based on standard practices in recommender systems and machine learning.

## Description of Functions and Classes

### Functions

1. **`readData(chunksize=20000)`**:
   - Reads the business and review data in chunks from JSON files.
   - Parameters: `chunksize` - Size of each data chunk.
   - Returns: Chunks of business and review data.

2. **`filterBusinesses(businesses)`**:
   - Filters the business dataset to include only relevant columns and businesses categorized as restaurants.
   - Parameters: `businesses` - DataFrame of business data.
   - Returns: Filtered DataFrame of businesses.

3. **`filterReviews(reviews)`**:
   - Filters the reviews dataset to include only user ID, business ID, stars, and date.
   - Parameters: `reviews` - DataFrame of review data.
   - Returns: Filtered DataFrame of reviews.

4. **`combineDataframes(businesses, reviews)`**:
   - Merges the business and review datasets on the business ID.
   - Parameters: `businesses`, `reviews` - DataFrames of business and review data.
   - Returns: Merged DataFrame.

5. **`createPivotTable(all_combined)`**:
   - Creates a pivot table from the combined dataset for collaborative filtering.
   - Parameters: `all_combined` - Merged DataFrame of businesses and reviews.
   - Returns: User-item pivot table.

6. **`calculate_similarity(user_item_matrix)`**:
   - Calculates the cosine similarity matrix from the user-item matrix.
   - Parameters: `user_item_matrix` - User-item pivot table.
   - Returns: Cosine similarity matrix.

7. **`predict_ratings(similarity, user_item_matrix, user_id)`**:
   - Predicts ratings for all items for a given user.
   - Parameters: `similarity`, `user_item_matrix`, `user_id`.
   - Returns: Predicted ratings for the user.

8. **`train_test_split_and_predict(data)`**:
   - Splits the data into train and test sets, predicts ratings, and returns the true and predicted ratings.
   - Parameters: `data` - User-item pivot table.
   - Returns: True and predicted ratings.

9. **`evaluate_performance(data)`**:
   - Evaluates the performance of the collaborative filtering algorithm.
   - Parameters: `data` - User-item pivot table.
   - Returns: RMSE and MAE values.

10. **`trainCollabFiltering(chunksize)`**:
    - Orchestrates the collaborative filtering training process.
    - Parameters: `chunksize` - Size of data chunks.
    - Returns: Performance metrics (RMSE, MAE).

11. **`encodingData(dataset)`**:
    - Encodes the user and business IDs in the dataset.
    - Parameters: `dataset` - Combined DataFrame of businesses and reviews.
    - Returns: Encoded dataset.

12. **`trainNN(combinedDataset)`**:
    - Trains the neural network model on the combined dataset.
    - Parameters: `combinedDataset` - Combined DataFrame of businesses and reviews.
    - Returns: RMSE and MAE values.

13. **`trainANN(chunksize)`**:
    - Orchestrates the neural network training process.
    - Parameters: `chunksize` - Size of data chunks.
    - Returns: Combined DataFrame for ANN training.

14. **`main()`**:
    - Main function to run the project.

### Classes

1. **`EmbeddingLayer`**:
   - A class representing an embedding layer in the neural network.
   - Attributes:
     - `n_items`: Number of items (users or restaurants).
     - `n_factors`: Number of latent factors.
   - Method:
     - `__call__(self, x)`: Applies embedding and reshape operations to the input.

2. **`Recommender`**:
   - Builds the neural network model for the recommender system.
   - Parameters: `n_users`, `n_rests`, `n_factors`, `min_rating`, `max_rating`.
   - Returns: Compiled Keras model.

Each function and class plays a crucial role in the pipeline of reading, processing, and analyzing the dataset to build and evaluate the recommender system using two different methodologies.
