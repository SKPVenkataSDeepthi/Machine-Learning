#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
import numpy as np


# In[2]:


# Getting reviews with words that come under 5000
# most occurring words in the entire
# corpus of textual review data
vocab_size = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print(x_train[0])


# In[3]:


# Getting all the words from word_index dictionary
word_idx = imdb.get_word_index()

# Originally the index number of a value and not a key,
# hence converting the index as key and the words as values
word_idx = {i: word for word, i in word_idx.items()}

# again printing the review
print([word_idx[i] for i in x_train[0]])


# In[4]:


# Get the minimum and the maximum length of reviews
print("Max length of a review:: ", len(max((x_train+x_test), key=len)))
print("Min length of a review:: ", len(min((x_train+x_test), key=len)))


# In[5]:


from tensorflow.keras.preprocessing import sequence

# Keeping a fixed length of all reviews to max 400 words
max_words = 400

x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)

x_valid, y_valid = x_train[:64], y_train[:64]
x_train_, y_train_ = x_train[64:], y_train[64:]


# In[6]:


# Defining the vocabulary size and embedding length
vocab_size = 10000  
embd_len = 100 

# Defining GRU model
gru_model = Sequential(name="GRU_Model")
gru_model.add(Embedding(vocab_size,
						embd_len,
						input_length=max_words))
gru_model.add(GRU(128,
				activation='tanh',
				return_sequences=False))
gru_model.add(Dense(1, activation='sigmoid'))

# Printing the Summary
print(gru_model.summary())

# Compiling the model
gru_model.compile(
	loss="binary_crossentropy",
	optimizer='adam',
	metrics=['accuracy']
)

# Training the GRU model
history2 = gru_model.fit(x_train_, y_train_,
						batch_size=64,
						epochs=5,
						verbose=1,
						validation_data=(x_valid, y_valid))

# Printing model score on test data
print()
print("GRU model Score---> ", gru_model.evaluate(x_test, y_test, verbose=0))


# In[7]:


pip install textblob


# In[8]:


import nltk
nltk.download('punkt')


# In[9]:


import requests
from textblob import TextBlob

def get_movie_sentiment_and_info(review_text, movie_title, api_key):
    # Analyze the sentiment of the review
    sentiment = analyze_sentiment(review_text)
    
    # Fetch movie information
    movie_info = fetch_movie_info(movie_title, api_key)
    
    return sentiment, movie_info

def analyze_sentiment(review_text):
    # Using TextBlob to perform sentiment analysis
    analysis = TextBlob(review_text)
    # Determine if sentiment is positive, neutral, or negative
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

def fetch_movie_info(movie_title, api_key):
    # Define the base URL for the OMDb API
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    
    try:
        # Send a request to the OMDb API
        response = requests.get(url)
        
        # Check if the response was successful
        if response.status_code == 200:
            data = response.json()
            if data['Response'] == 'True':
                # Extract relevant movie information
                movie_info = {
                    "Title": data.get("Title"),
                    "Director": data.get("Director"),
                    "Year": data.get("Year"),
                    "Genre": data.get("Genre"),
                    "Plot": data.get("Plot"),
                    "Actors": data.get("Actors"),
                    "IMDb Rating": data.get("imdbRating"),
                }
            else:
                movie_info = {"Error": data.get("Error", "Movie not found!")}
        else:
            movie_info = {"Error": f"Error {response.status_code}: Unable to fetch data!"}
    except requests.exceptions.RequestException as e:
        movie_info = {"Error": f"Request failed: {str(e)}"}
    
    return movie_info

def main():
    movie_title = input("Enter movie title: ")
    review_text = input("Enter your review: ")
    api_key = "5134ecb8"  
    sentiment, movie_info = get_movie_sentiment_and_info(review_text, movie_title, api_key)
    
    print(f"Sentiment: {sentiment}")
    print("Movie Information:")
    if "Error" in movie_info:
        print(movie_info["Error"])
    else:
        for key, value in movie_info.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()


# Reference:
# https://www.geeksforgeeks.org/sentiment-analysis-with-an-recurrent-neural-networks-rnn/?ref=lbp

# In[ ]:




