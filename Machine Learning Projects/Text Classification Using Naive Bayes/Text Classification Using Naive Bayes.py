#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the dataset
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
newsgroups


# In[3]:


text_categories = newsgroups.target_names
text_categories

# Print the text categories
print("Text Categories:")
for i, category in enumerate(text_categories):
    print(f"{i}: {category}")

print("Number of unique classes: {}".format(len(text_categories)))


# In[4]:


# Split the dataset
train_data, test_data, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)


# In[5]:


# Create a pipeline that combines the TfidfVectorizer and the MultinomialNB classifier
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
model


# In[6]:


# Train the model
model.fit(train_data, y_train)


# In[7]:


# Make predictions
y_pred = model.predict(test_data)
y_pred


# In[8]:


# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[9]:


# Print classification report
print(metrics.classification_report(y_test, y_pred, target_names=newsgroups.target_names))


# In[10]:


# Compute confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=newsgroups.target_names, yticklabels=newsgroups.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[11]:


# Print train and test data with their respective categories
print("\nTrain Data Categories:")
for i in range(len(train_data)):
    print(f"Document {i}: Category - {text_categories[y_train[i]]}")

print("\nTest Data Categories:")
for i in range(len(test_data)):
    print(f"Document {i}: Category - {text_categories[y_test[i]]}")


# In[ ]:




