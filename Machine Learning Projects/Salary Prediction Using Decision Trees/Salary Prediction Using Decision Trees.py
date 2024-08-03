#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Data Science Salary 2021 to 2023.csv')

# Display the first few rows of the dataset
print(df.head())


# In[2]:


df.info()


# In[3]:


# Convert 'experience_level' to categorical codes
df['experience_level'] = df['experience_level'].astype('category').cat.codes

# Convert 'employment_type' to categorical codes
df['employment_type'] = df['employment_type'].astype('category').cat.codes

# Convert 'company_location' to categorical codes
df['company_location'] = df['company_location'].astype('category').cat.codes

# Convert 'company_size' to categorical codes
df['company_size'] = df['company_size'].astype('category').cat.codes

# Convert 'job_title' to categorical codes 
df['job_title'] = df['job_title'].astype('category').cat.codes

# Convert 'salary_currency' to numeric
# If 'USD' is the only currency, it can be represented as a constant or removed if redundant
df['salary_currency'] = (df['salary_currency'] == 'USD').astype(int)

# Ensure 'work_year', 'salary', and 'salary_in_usd' are numeric
df['work_year'] = df['work_year'].astype(int)
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
df['salary_in_usd'] = pd.to_numeric(df['salary_in_usd'], errors='coerce')


# In[4]:


df.info()


# In[5]:


# Import the libraries 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[6]:


print(df.columns)


# In[7]:


# Prepare data
X = df[['work_year', 'experience_level', 'employment_type','job_title','salary_currency','salary_in_usd','company_location','company_size']]  # Features
y =df['salary']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Initialize and train the model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)


# In[9]:


# Make predictions
y_pred = model.predict(X_test)
y_pred


# In[10]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[11]:


# Tune hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")


# In[12]:


# Feature importance 
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()


# In[13]:


# Visualize the decision tree 
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, rounded=True, class_names=['Not High Salary', 'High Salary'])
plt.title('Decision Tree Visualization')
plt.show()


# In[14]:


# Visualize the decision tree 
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree with increased figure size and font size
plt.figure(figsize=(30, 15))  # Increase figure size
plot_tree(model, 
          filled=True, 
          feature_names=X.columns, 
          rounded=True, 
          precision=2,  # Set precision to 2 decimal places
          fontsize=20)  # Increase font size
plt.title('Decision Tree Visualization')
plt.show()


# In[ ]:




