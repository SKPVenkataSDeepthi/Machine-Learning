# Book Recommendation Engine Using K-Nearest Neighbors (KNN)

This project demonstrates how to build a book recommendation algorithm using K-Nearest Neighbors (KNN) with the Book-Crossings dataset. The goal is to recommend books similar to a given book based on user ratings.

## Dataset

The dataset used is the Book-Crossings dataset, which includes:
- 1.1 million ratings of 270,000 books by 90,000 users.

## Features

The model uses the following features:
- Book ratings

## Steps

1. **Data Preprocessing**
   - Imported and cleaned the dataset.
   - Removed users with less than 200 ratings and books with less than 100 ratings to ensure statistical significance.

2. **Model Training**
   - Utilized `NearestNeighbors` from `sklearn.neighbors` to build a KNN model.
   - Trained the model to find similar books based on user ratings.

3. **Recommendation Function**
   - Created a function `get_recommends(book_title)` that takes a book title as input and returns a list of 5 similar books with their distances from the input book.

4. **Testing**
   - Verified the function with sample input and ensured that it returns the correct format and recommendations.

## Dependencies
To run this project, you need the following Python packages:
numpy
pandas
scikit-learn
matplotlib (optional for visualization)
