# Salary Prediction Using Decision Trees

This project involves predicting salaries using a Decision Tree Regressor. The goal is to build a model that can accurately predict salaries based on various features. The project also includes hyperparameter tuning to optimize the model's performance.

## Project Overview

The objective of this project is to use a Decision Tree Regressor to predict salaries based on a given dataset. The project includes:
- Data preprocessing
- Model training
- Model evaluation
- Hyperparameter tuning
- Visualization of the decision tree

## Dataset

The dataset used for this project should include various features that can influence salary. Ensure your dataset is in a CSV format and split into training and testing sets.


## Model Evaluation

The model is evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics. Initial model evaluation is performed before hyperparameter tuning.

## Hyperparameter Tuning

Hyperparameter tuning is done using `GridSearchCV` to find the best parameters for the Decision Tree Regressor. The following parameters are tuned:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

## Decision Tree Visualization

The decision tree is visualized using `plot_tree` from `sklearn.tree`. The tree is plotted with increased figure size and font size to improve readability.

## Results

After hyperparameter tuning, the model's performance is re-evaluated and compared to the initial results. The best hyperparameters are printed and the final decision tree is visualized.
