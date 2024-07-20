## Cat and Dog Image Classifier ##

This project was completed as part of a Machine Learning with Python course. The goal of this project is to classify images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow 2.0 and Keras. 

## Project Overview

The dataset used in this project consists of images of cats and dogs, divided into training, validation, and test sets. The project involves creating a convolutional neural network to classify these images with at least 63% accuracy (extra credit if the accuracy exceeds 70%).

## Instructions

1. **Import Libraries**: The first step is to import the necessary libraries for the project.

2. **Download Data**: The dataset is downloaded and key variables are set.

3. **Image Generators**: Set up `ImageDataGenerator` instances for the training, validation, and test datasets. Rescale pixel values from 0-255 to 0-1.

4. **Plot Images**: Visualize sample images from the training dataset.

5. **Data Augmentation**: Enhance the training data with random transformations to prevent overfitting.

6. **Model Creation**: Build a CNN model using the Keras Sequential API. The model should include Conv2D and MaxPooling2D layers, followed by a fully connected layer.

7. **Model Training**: Train the model using the `fit` method, with appropriate arguments for epochs and validation data.

8. **Visualization**: Analyze the model's performance by visualizing accuracy and loss.

9. **Prediction**: Use the trained model to classify images from the test dataset. Display the images with predicted probabilities.

10. **Challenge Evaluation**: Run the final cell to check if the model meets the required accuracy.

## Files

- `cat_vs_dog_classification.ipynb`: Jupyter notebook containing all code cells and instructions.

## Requirements

- TensorFlow 2.0
- Keras
- Other standard Python libraries for data handling and visualization
