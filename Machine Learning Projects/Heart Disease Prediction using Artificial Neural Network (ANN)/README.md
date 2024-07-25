## Heart Disease Prediction using Artificial Neural Network ##
This repository contains a project to predict whether a person has heart disease or not using an Artificial Neural Network (ANN). The project uses the TensorFlow and Keras libraries to build and train the ANN model, and the dataset used is the publicly available heart disease dataset.

## Introduction to Deep Learning and ANN
Deep Learning is a technology that mimics a human brain in the sense that it consists of multiple neurons with multiple layers like a human brain. The network so formed consists of an input layer, an output layer, and one or more hidden layers. The network tries to learn from the data that is fed into it and then performs predictions accordingly.

The most basic type of neural network is the Artificial Neural Network (ANN). The ANN does not have any special structure, it just comprises multiple neural layers to be used for prediction.

## Dataset
The dataset used in this project is heart.csv, which contains various health metrics of individuals. The target variable is "target", which indicates the presence (1) or absence (0) of heart disease.

## Model Overview
The model is an Artificial Neural Network (ANN) with the following architecture:

* Input Layer: 13 neurons (corresponding to the 13 features in the dataset)
* Hidden Layer 1: 8 neurons with ReLU activation
* Hidden Layer 2: 14 neurons with ReLU activation
* Output Layer: 1 neuron with Sigmoid activation (for binary classification)
## Steps Involved
* Import Libraries: Import necessary libraries including TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
* Load and Prepare the Dataset: Load the dataset and separate the features and target variable.
* Split Data: Split the data into training and testing sets.
* Scale the Data: Scale the features using standard scaling.
* Build the ANN Model: Create a Sequential model and add layers (input layer, hidden layers, and output layer) with appropriate activation functions.
* Compile the Model: Compile the model with optimizer, loss function, and evaluation metrics.
* Train the Model: Train the model using the training data.
* Make Predictions: Predict the outcomes for the test set and the entire dataset.
* Calculate Percentages: Calculate the percentage of individuals predicted to have and not to have heart disease.
* Plot Results: Plot a pie chart showing the percentages of individuals with and without heart disease.
* Evaluate Model: Create and visualize the confusion matrix for the test set predictions.
## Results
The model predicts the presence of heart disease with an accuracy of approximately 85%. The confusion matrix and percentage plots provide a detailed view of the model's performance.
## Conclusion
This project demonstrates the use of Artificial Neural Networks for predicting heart disease. It provides a comprehensive overview of the steps involved in data preparation, model building, training, prediction, and evaluation.
