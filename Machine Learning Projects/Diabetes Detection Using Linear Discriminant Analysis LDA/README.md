# Diabetes Detection Using Linear Discriminant Analysis (LDA)

This project demonstrates the use of Linear Discriminant Analysis (LDA) to detect diabetes based on the Glucose and BMI features from a diabetes dataset. The goal is to classify whether a patient has diabetes or not using these features.

## Dataset

The dataset used is `diabetes2.csv`, which includes the following columns:
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (target variable: 0 for non-diabetic, 1 for diabetic)

## Features

For simplicity, the model uses the following features:
- Glucose
- BMI

## Steps

1. **Data Preprocessing**
   - Loaded and inspected the dataset.
   - Encoded the target variable (`Outcome`) as categorical.
   - Standardized the features using `StandardScaler`.

2. **Model Training**
   - Split the data into training and testing sets.
   - Trained the LDA model on the training set.

3. **Model Evaluation**
   - Evaluated the model's performance on the test set using accuracy, precision, recall, and F1-score.
   - Generated and visualized the confusion matrix.
   - Plotted the ROC curve and calculated the AUC.

4. **Visualization**
   - Created decision regions plot to visualize how the LDA model separates the classes based on Glucose and BMI.

## Results

- **Training Accuracy:** 76.5%
- **Test Accuracy:** 79.0%

The confusion matrix and classification report provide detailed insights into the model's performance. The ROC curve shows the trade-off between true positive rate and false positive rate.

## Confusion Matrix

![Confusion Matrix](path/to/confusion_matrix_plot.png)

## ROC Curve

![ROC Curve](path/to/roc_curve_plot.png)

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
