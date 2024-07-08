#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[2]:


import pandas as pd

dataset = "diabetes2.csv"
df = pd.read_csv(dataset)
df.head()


# In[3]:


print(df.dtypes)


# In[4]:


import pandas as pd

dataset = pd.read_csv('diabetes2.csv')
dataset['Outcome'] = dataset['Outcome'].map({0: 'ND', 1: 'D'})
outcome_counts = dataset['Outcome'].value_counts()
print(outcome_counts)


# In[5]:


x = dataset.iloc[:, 2:20].values  # Select columns 2 to 19 (adjust as needed)
y = dataset['Outcome'].values  # Assuming 'Outcome' is your target variable

# Print the shape of x and y for verification
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[6]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode categorical variable 'Outcome' to numeric format
le = LabelEncoder()
dataset['Outcome'] = le.fit_transform(dataset['Outcome'])

# Separate features (x) and target variable (y)
x = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dataset['Outcome']

# Instantiate StandardScaler
scaler = StandardScaler()

# Fit and transform the features (x)
x_scaled = scaler.fit_transform(x)

# Convert scaled data (numpy array) back to a DataFrame
scaled_df = pd.DataFrame(x_scaled, columns=x.columns)

# Print the scaled data DataFrame
print("Scaled Data Table:")
print(scaled_df.head())  # Display the first few rows


# In[7]:


#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state =0)


# In[8]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(x_train,y_train)
clf.score(x_train,y_train)


# In[9]:


#Prediction
y_pred = clf.predict(x_test)
y_pred


# In[10]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[11]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[12]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Reds', fmt='g', cbar=False)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[13]:


from sklearn.metrics import classification_report, roc_curve, auc

# Predict probabilities for the test set
y_prob = clf.predict_proba(x_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap


# Select two features for simplicity
features = ['Glucose','BMI']
x = df[features]
y = df['Outcome']

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)

# Train the LDA model
clf = LinearDiscriminantAnalysis()
clf.fit(x_train, y_train)

# Evaluate the model
print(f"Training accuracy: {clf.score(x_train, y_train)}")
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))

# Create a mesh grid for plotting decision regions
x_min, x_max = x_scaled[:, 0].min() - 1, x_scaled[:, 0].max() + 1
y_min, y_max = x_scaled[:, 1].min() - 1, x_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the class for each point in the mesh grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green')))

# Plot the training points
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=x_scaled[y == cl, 0], y=x_scaled[y == cl, 1],
                alpha=0.8, c=ListedColormap(('red', 'green'))(idx),
                marker='o', label=cl, edgecolor='black')
plt.xlabel('Glucose (standardized)')
plt.ylabel('BMI (standardized)')
plt.legend(loc='upper left')
plt.title('LDA Decision Regions for Diabetes Dataset')
plt.show()


# In[ ]:




