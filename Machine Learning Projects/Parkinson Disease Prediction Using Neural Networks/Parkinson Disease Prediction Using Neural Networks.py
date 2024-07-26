#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import shap


# In[2]:


# Function to preprocess input data
def preprocess_input(input_data, scaler):
    return scaler.transform(input_data)


# In[3]:


# Function to predict Parkinson's disease
def predict_parkinsons(input_data, model, scaler, threshold=0.5):
    # Preprocess the input data
    input_data_scaled = preprocess_input(input_data, scaler)
    
    # Predict the probabilities
    y_prob = model.predict(input_data_scaled)
    
    # Convert probabilities to binary class labels
    y_pred = (y_prob > threshold).astype(int)
    
    return y_pred


# In[4]:


# Load dataset
data = pd.read_csv('parkinsons.csv')
data.head()


# In[5]:


# Preprocess the dataset
X = data.drop(columns=['name', 'status'])
y = data['status']


# In[6]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scaler
X_train
X_test


# In[8]:


# Define the model with Dropout and/or L2 Regularization
def create_model(optimizer='adam', activation='relu', use_dropout=False, use_l2=False):
    model = Sequential()
    reg = l2(0.01) if use_l2 else None

    model.add(Dense(128, input_dim=X_train.shape[1], activation=activation, kernel_regularizer=reg))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(64, activation=activation, kernel_regularizer=reg))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(32, activation=activation, kernel_regularizer=reg))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer_instance = Adam() if optimizer == 'adam' else RMSprop()
    model.compile(optimizer=optimizer_instance, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[9]:


# Re-train the model with best parameters
best_model = create_model(optimizer='adam', activation='relu', use_dropout=True, use_l2=True)
history = best_model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=40, verbose=0)


# In[10]:


# Plot training history (accuracy and loss)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.show()


# In[11]:


# Predict probabilities on the test set
y_prob = best_model.predict(X_test)
y_prob


# In[12]:


# Convert probabilities to binary class labels
y_test_pred = (y_prob > 0.5).astype(int).ravel()
y_test = y_test.astype(int).ravel()


# In[13]:


# Print classification report
print(classification_report(y_test, y_test_pred, target_names=['No Parkinson\'s', 'Parkinson\'s']))


# In[14]:


# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_prob.ravel())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[15]:


# SHAP Values
explainer = shap.KernelExplainer(best_model.predict, X_train)
shap_values = explainer.shap_values(X_test)

# Plot SHAP values
shap.summary_plot(shap_values, X_test)


# In[16]:


# Extract a sample row from the dataset
# Take the first row
sample_row = X.iloc[0]

# Convert the row to a numpy array
new_sample = sample_row.values.reshape(1, -1)

# Preprocess and predict using the sample
new_sample_scaled = scaler.transform(new_sample)
new_sample_pred = best_model.predict(new_sample_scaled)
if new_sample_pred == 1:
    print("Parkinson's detected")
else:
    print("No Parkinson's")


# In[17]:


# Define and compile the model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Standardize the entire feature set
X_scaled = scaler.transform(X)

# Standardize the entire feature set
X_scaled = scaler.transform(X)

# Predict the entire dataset
predictions = (model.predict(X_scaled) > 0.5).astype(int).ravel()

# Add predictions to the original dataset
data['predictions'] = predictions

# Print the number of samples with and without Parkinson's disease
num_with_disease = np.sum(predictions)
num_without_disease = len(predictions) - num_with_disease
print(f"Number of samples predicted to have Parkinson's: {num_with_disease}")
print(f"Number of samples predicted not to have Parkinson's: {num_without_disease}")

# Plot the results
labels = ['With Parkinson\'s', 'Without Parkinson\'s']
sizes = [num_with_disease, num_without_disease]
colors = ['#ff9999','#66b3ff']
explode = (0.1, 0)  # explode 1st slice

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.title('Predicted Distribution of Parkinson\'s Disease')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[ ]:




