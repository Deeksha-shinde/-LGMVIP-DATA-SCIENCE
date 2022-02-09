#!/usr/bin/env python
# coding: utf-8

# # LGM-VIP Data Science Internship Programme
# ## Beginner Level Task-1 Iris Flower Classification 
# ### Submitted by : Deeksha Shinde

# # Importing Libraries:

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# read the csv file 

iris = pd.read_csv('D:/IRIS.csv')


# In[4]:


iris


# # Read Dataset
# First 5 records

# In[5]:


iris.head()


# In[6]:


iris.tail()


# # Size of Dataset

# In[7]:


iris.shape


# # Find null value in datase

# In[8]:


iris.isna().sum()


# # Find duplicates in dataset

# In[9]:


iris.duplicated().sum()


# # Description of Dataset

# In[10]:


iris.describe()


# # Datatypes

# In[11]:


iris.dtypes


# # Information of dataset

# In[12]:


iris.info()


# # All the columns

# In[13]:


iris.columns


# In[14]:


print(iris.groupby(["species"]).size())


# In[15]:


a = len(iris[iris['species'] == 'Iris-versicolor'])
print("No of Versicolor in Dataset:",a)
b = len(iris[iris['species'] == 'Iris-virginica'])
print("No of Virginica in Dataset:",b)
c = len(iris[iris['species'] == 'Iris-setosa'])
print("No of Setosa in Dataset:",c)


# # Data Visualization/Exploratory Data Analysis

# In[16]:


iris.species.value_counts()


# # Correlation Matrix

# In[17]:


corrmat = iris.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heatmap
sns.heatmap(iris[top_corr_features].corr(), annot=True, cmap="RdYlGn")


# # Pieplot

# In[18]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()


# # Violinplot

# In[19]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='petal_length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='petal_width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='sepal_width',data=iris)


# # Scatter plot
# 

# In[20]:


plt.figure(figsize=(12,10))
plt.scatter(iris.petal_length[iris.species=="Iris-setosa"],iris.petal_width[iris.species=="Iris-setosa"],c="red",label="Iris-setosa",marker='*')
plt.scatter(iris.petal_length[iris.species=="Iris-versicolor"],iris.petal_width[iris.species=="Iris-versicolor"],c='b',label="Iris-versicolor",marker='*')
plt.scatter(iris.petal_length[iris.species=="Iris-virginica"],iris.petal_width[iris.species=="Iris-virginica"],c="green",label="Iris-virginica",marker='*')

plt.xlabel("Petal length")
plt.ylabel("Petal Width")
plt.legend()


# # Pair plot

# In[21]:


sns.pairplot(iris, hue='species')


# In[22]:


sns.pairplot(data=iris, diag_kind='kde')
plt.show()


# # Histogram

# In[23]:


iris.hist()
plt.show()


# #   Count plot 

# In[24]:


sns.countplot(iris['species'])


# # Heatmap

# In[25]:


sns.heatmap(iris.corr(),square= True)


# # Training the Model

# ## Data splitting into train and test dataset

# In[26]:


x = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris[['species']]


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0, stratify = y)


# ## DecisionTree Classifier Model 

# In[28]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=2)


# In[29]:



#Fit Model
model.fit(X_train, y_train)


# ## Make predictions

# In[30]:


y_pred = model.predict(X_test)
y_pred


# ## Accuracy of Model 

# In[31]:


model.score(X_test, y_test)


# In[32]:


model.score(X_train, y_train)


# In[ ]:




