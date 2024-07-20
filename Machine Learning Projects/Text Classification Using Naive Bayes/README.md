# Text Classification with 20 Newsgroups Dataset

This project demonstrates a complete workflow for text classification using the 20 Newsgroups dataset and a Naive Bayes classifier. It provides a comprehensive guide to building, training, evaluating, and visualizing a text classification model.

## Project Overview

The project includes the following steps:
1. **Data Loading and Preprocessing**
2. **Model Training**
3. **Model Evaluation**
4. **Results Visualization**

## Steps

### 1. Import Libraries

Necessary libraries are imported for machine learning and data visualization:
- Scikit-learn for machine learning tasks.
- Matplotlib and Seaborn for visualization.

### 2. Load the Dataset

The `fetch_20newsgroups` function from Scikit-learn is used to load the 20 Newsgroups dataset, which contains text data categorized into 20 different topics.

### 3. Print Categories and Number of Classes

Retrieve and print the text categories to understand the different topics present in the dataset. Also, print the number of unique classes to see how many distinct categories are present.

### 4. Split the Dataset

The dataset is split into training and testing sets using `train_test_split` to ensure that the model can be evaluated on unseen data.

### 5. Create a Pipeline

A pipeline is created using `make_pipeline` that combines:
- `TfidfVectorizer`: Transforms the text data into TF-IDF (Term Frequency-Inverse Document Frequency) features.
- `MultinomialNB`: A Naive Bayes classifier suitable for classification with discrete features (e.g., word counts).

### 6. Train the Model

The pipeline model is trained on the training data, learning the parameters of both the vectorizer and the classifier.

### 7. Make Predictions

The trained model is used to predict the categories of the test data.

### 8. Evaluate the Model

Model evaluation includes:
- Calculating accuracy: The proportion of correctly classified documents.
- Printing a classification report: Includes precision, recall, and F1-score for each category.
- Computing and visualizing the confusion matrix: Shows the counts of true positive, false positive, true negative, and false negative predictions for each category.

### 9. Print Train and Test Data with Categories

Print the categories of documents in both the training and test datasets to understand the distribution and examples of data points.

## Files

- `text_classification.ipynb`: Jupyter notebook containing the code and instructions for the project.

## Requirements

- Python
- Libraries: Scikit-learn, Matplotlib, Seaborn (and any other dependencies required for data processing and visualization)

