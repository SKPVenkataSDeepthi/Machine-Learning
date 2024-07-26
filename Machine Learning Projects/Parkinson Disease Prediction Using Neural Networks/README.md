# Parkinson's Disease Prediction Using Neural Networks

This project aims to predict the presence of Parkinson's disease using machine learning techniques, specifically a neural network model. The model is trained on a dataset of voice recordings, where each sample is labeled as either having Parkinson's disease or not.

## Overview

Parkinson's disease is a neurodegenerative disorder that affects movement. Early detection is crucial for effective treatment. This project utilizes a neural network to classify voice recordings based on features extracted from the recordings. The neural network is trained to distinguish between samples from patients with Parkinson's disease and those without.

## Dataset

The dataset used for this project is the [Parkinson's Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinsons). It includes various features derived from voice recordings and is labeled with the presence or absence of Parkinson's disease.

## Model Architecture

The neural network model used in this project is a sequential model with the following architecture:

- **Input Layer**: 128 neurons, ReLU activation
- **Hidden Layer 1**: 64 neurons, ReLU activation
- **Hidden Layer 2**: 32 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation (for binary classification)

The model is compiled with the Adam optimizer and binary cross-entropy loss function. It is trained for 100 epochs with a batch size of 10.

## Requirements

To run this project, you'll need the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

