## Sentiment Analysis with Recurrent Neural Networks (RNN)
This project involves building a Sentiment Analysis system using Recurrent Neural Networks (RNNs) to analyze movie reviews from the IMDB dataset. RNNs are ideal for capturing the sequence of information in text data, making them suitable for tasks like sentiment analysis, time series prediction, and more. This project demonstrates the use of a Gated Recurrent Unit (GRU) model to classify the sentiment of movie reviews.

# Introduction
Recurrent Neural Networks (RNNs) are a class of neural networks that are effective for sequential data tasks. In this project, we use an RNN with a GRU layer to analyze the sentiment of movie reviews from the IMDB dataset. The model predicts whether a review is positive or negative based on the text.

# Dataset
We use the IMDB dataset, which contains 50,000 movie reviews labeled as positive or negative. The dataset is available in Keras and includes the load_data function, which allows us to easily load the data with a specified vocabulary size.

# Model Architecture
The model consists of the following layers:
* Embedding Layer: Converts integer sequences into dense vectors of fixed size.
* GRU Layer: Captures the sequential nature of the text data.
* Dense Layer: Outputs the final sentiment prediction.

# Movie Information and Sentiment Analysis
In addition to sentiment analysis, this project includes a feature to fetch movie information using the OMDb API. Given a movie title and a review, the script analyzes the sentiment of the review and fetches relevant movie information.

# Installation
To run this project, you need to have Python installed along with the following libraries:
* TensorFlow
* Keras
* TextBlob
* requests

# Results
The GRU model achieves good accuracy on the test set. It is effective at capturing the sentiment of reviews. Additionally, the script provides detailed information about the movie, including the director, year, genre, plot, and IMDb rating.

# Ref. 
GeeksForGeeks. 
