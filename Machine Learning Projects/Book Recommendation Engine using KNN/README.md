# Book Recommendation Engine Using K-Nearest Neighbors (KNN)

## Project Overview

The Book Recommendation Engine project utilizes the K-Nearest Neighbors (KNN) algorithm to build a book recommendation system. By analyzing user ratings from the Book-Crossings dataset, this project aims to recommend books that are similar to a given book. The recommendation is based on the ratings provided by users, leveraging the KNN algorithm to find and suggest books with similar rating patterns.

The primary objective of this project is to provide users with personalized book recommendations, enhancing their reading experience by suggesting books that are likely to be of interest based on their past preferences.

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
