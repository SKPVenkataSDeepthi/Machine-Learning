#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas scikit-learn nltk


# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import nltk #Natural Langunage Tool Kit
from nltk.corpus import stopwords #StopWords are commonly used words
from nltk.stem import SnowballStemmer
import re


# In[3]:


nltk.download('stopwords')


# In[4]:


df = pd.read_csv("spam_ham_dataset.csv")
df


# In[5]:


df.info()


# In[6]:


#Data Preprocessing

def preprocessingText(text):
    text = re.sub('[^a-zA-Z]',' ',text)
    text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]
    return ''.join(words)
                  


# In[7]:


# Apply preprocessing to the text data
df['text'] = df['text'].apply(preprocessingText)
df['text']


# In[8]:


# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['text'])
X


# In[9]:


# Convert labels to binary (0 for ham, 1 for spam)
y = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
y


# In[10]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)


# In[11]:


# Make predictions
y_pred = svm.predict(X_test)
y_pred


# In[12]:


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




