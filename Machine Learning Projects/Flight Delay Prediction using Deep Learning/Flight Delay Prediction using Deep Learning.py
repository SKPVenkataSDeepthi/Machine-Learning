#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[2]:


# Load the data
data = pd.read_csv('flight_data_2018_to_2022.csv')
data.head()


# In[3]:


print(data.columns)


# In[4]:


# Check for missing values
missing_data = data.isna().sum()
print("Missing values in each column:")
print(missing_data)


# In[5]:


# Option 1: Fill missing values with median or mean
data.fillna({
    'AirTime': data['AirTime'].median(),
    'Distance': data['Distance'].median(),
    'Origin': 'UNKNOWN',
    'Dest': 'UNKNOWN',
    'ArrDelay': data['ArrDelay'].median(),
    'FlightDate': pd.Timestamp('1900-01-01')
}, inplace=True)


# In[6]:


# Verify that there are no more missing values
missing_data = data.isna().sum()
print("\nMissing values after handling:")
print(missing_data)


# In[7]:


# Proceed if the dataset is not empty after handling missing values
if data.empty:
    raise ValueError("The dataset is empty after handling missing values. Please check your data.")

# Plot average arrival delay by origin airport
avg_delay_by_origin = data.groupby('Origin')['ArrDelay'].mean().reset_index()
bar_plot_origin = px.bar(avg_delay_by_origin, x='Origin', y='ArrDelay', title='Average Arrival Delay by Origin Airport')
bar_plot_origin.update_layout(xaxis_title='Origin Airport', yaxis_title='Average Arrival Delay')
bar_plot_origin.show()

# Plot average arrival delay by destination airport
avg_delay_by_dest = data.groupby('Dest')['ArrDelay'].mean().reset_index()
bar_plot_dest = px.bar(avg_delay_by_dest, x='Dest', y='ArrDelay', title='Average Arrival Delay by Destination Airport')
bar_plot_dest.update_layout(xaxis_title='Destination Airport', yaxis_title='Average Arrival Delay')
bar_plot_dest.show()

# Plot average delay by month
data['FlightDate'] = pd.to_datetime(data['FlightDate'])
avg_delay_month = data.groupby(data['FlightDate'].dt.month)['ArrDelay'].mean().reset_index()
fig_month = px.bar(avg_delay_month, x='FlightDate', y='ArrDelay', labels={'FlightDate': 'Month', 'ArrDelay': 'Average Delay'}, title='Average Delay by Month')
fig_month.update_traces(marker_color='skyblue')
fig_month.show()


# In[8]:


# Prepare data for training
X = data[['AirTime', 'Distance', 'Origin', 'Dest']]
y = data['ArrDelay']


# In[9]:


# Encode categorical variables
label_encoder_origin = LabelEncoder()
label_encoder_dest = LabelEncoder()
X['Origin'] = label_encoder_origin.fit_transform(X['Origin'])
X['Dest'] = label_encoder_dest.fit_transform(X['Dest'])


# In[10]:


# Check if the dataset is empty after encoding
if X.empty or y.empty:
    raise ValueError("The dataset is empty after encoding categorical variables. Please check your data.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the training set is empty
if X_train.shape[0] == 0 or y_train.shape[0] == 0:
    raise ValueError("The training set is empty after splitting the data. Please check your data and parameters.")


# In[11]:


# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[12]:


# Build and train the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)


# In[13]:


# Evaluate the model
score, mse = model.evaluate(X_test, y_test, verbose=0)
print(f'Model mean squared error: {mse:.2f}')


# In[14]:


import numpy as np

# Assume 'df' is your DataFrame containing flight information
# Make sure 'Flight_Number_Marketing_Airline' is in 'df'
# The following is a mock implementation of how you might include this in the prediction:

# Real-time Prediction
air_time = float(input("Enter Air Time in minutes: "))
distance = float(input("Enter Distance in miles: "))
origin = input("Enter Origin Airport Code: ")
dest = input("Enter Destination Airport Code: ")

# Prepare user input for prediction
user_input = np.array([[air_time, distance, label_encoder_origin.transform([origin])[0], label_encoder_dest.transform([dest])[0]]])
user_input_scaled = scaler.transform(user_input)

# Make prediction
prediction = model.predict(user_input_scaled)[0][0]

print(f"The predicted arrival delay is {prediction:.2f} minutes.")

