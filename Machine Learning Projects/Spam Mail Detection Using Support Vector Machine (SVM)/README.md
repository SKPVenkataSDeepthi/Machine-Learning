# Spam Mail Detection System

This project aims to build a spam mail detection system using a Support Vector Machine (SVM) classifier to effectively classify emails as spam or ham. The project involves several key steps, including data preprocessing, feature extraction, model training, and evaluation.

## Project Overview

The goal of this project is to classify emails as either spam or ham using an SVM classifier. The following steps outline the approach taken to develop this system:

## Steps Involved

1. **Import Libraries**: Import necessary Python libraries for data processing, text handling, and machine learning.

2. **Download NLTK Stopwords**: Ensure that the NLTK stopwords data files are available for text preprocessing.

3. **Load Dataset**: Load the dataset from a CSV file and perform an initial inspection of the first few rows to understand its structure.

4. **Data Preprocessing Function**:
    - **Text Cleaning**: Remove non-alphabetic characters from the text.
    - **Lowercasing**: Convert all text to lowercase.
    - **Tokenization and Stopword Removal**: Split the text into individual words and remove common stopwords.
    - **Stemming**: Use the SnowballStemmer to reduce words to their root form.
    - **Joining Words**: Combine the processed words back into a single string.

5. **Apply Preprocessing to the Dataset**: Apply the preprocessing function to the text data in the dataset to prepare it for feature extraction.

6. **Feature Extraction using TF-IDF**:
    - **TF-IDF Vectorization**: Transform the text data into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) method. Limit the number of features to 3000.

7. **Convert Labels to Binary**: Convert the labels in the dataset to binary format where 'spam' is 1 and 'ham' is 0.

8. **Split Data into Training and Testing Sets**: Split the data into training (80%) and testing (20%) sets, ensuring reproducibility with a fixed `random_state`.

9. **Train SVM Model**: Train an SVM model with a linear kernel using the training data.

10. **Make Predictions**: Use the trained SVM model to make predictions on the test data.

11. **Evaluate the Model**: Print the accuracy and classification report (precision, recall, F1-score) to evaluate the performance of the model.

## Files

- `spam_mail_detection.ipynb`: Jupyter notebook containing the code for the project and instructions.

## Requirements

- Python
- Libraries: Pandas, NumPy, Scikit-Learn, NLTK, and any other dependencies required for data processing and model training.

