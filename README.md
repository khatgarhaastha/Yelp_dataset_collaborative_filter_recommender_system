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
