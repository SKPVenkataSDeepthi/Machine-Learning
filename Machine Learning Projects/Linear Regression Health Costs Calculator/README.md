# Linear Regression Health Costs Calculator

This project was completed as part of a Machine Learning course. The objective is to predict healthcare costs using a regression algorithm. The dataset contains information about various individuals, including their healthcare costs. The challenge involves creating a model that predicts these costs based on new data.

## Project Overview

In this project, we use linear regression to predict healthcare expenses. The dataset is split into training and testing subsets, and the model is evaluated based on its ability to predict healthcare costs with a Mean Absolute Error (MAE) under $3500.

## Instructions

1. **Import Libraries and Data**: Import the necessary libraries and load the dataset into the environment.

2. **Data Preprocessing**: Convert categorical data into numerical values. Split the dataset into training (80%) and testing (20%) subsets.

3. **Label Extraction**: Remove the "expenses" column from the datasets to create `train_labels` and `test_labels`. These labels are used for training and evaluating the model.

4. **Model Creation and Training**: Create a linear regression model and train it using the `train_dataset`.

5. **Model Evaluation**: Evaluate the model using the test dataset. Ensure that the Mean Absolute Error (MAE) is under $3500.

6. **Prediction and Visualization**: Use the model to predict healthcare expenses on the test dataset and visualize the results.

## Files

- `health_costs_calculator.ipynb`: Jupyter notebook containing the code for the project and instructions.

## Requirements

- Python
- Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib (and any other dependencies required for data handling and visualization)
