#!/usr/bin/env python
# coding: utf-8

# # TASK 1 - BEGINNER LEVEL
# # Iris Flowers Classification ML Project

# In[1]:


from sklearn import datasets
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score


# In[2]:


iris = datasets.load_iris()
digits = datasets.load_digits()


# In[3]:


print(digits.data)


# In[4]:


print(digits.target)


# In[5]:


print(iris.data)


# In[6]:


print(iris.target)


# In[7]:


print(iris.target_names)


# In[8]:


iris_data = iris.data #variable for array of the data
iris_target = iris.target #variable for array of the labels


# In[9]:


import matplotlib.pyplot as plt
plt.show()
sns.boxplot(data = iris_data,width=0.5,fliersize=5)
sns.set(rc={'figure.figsize':(1,10)})


# In[10]:


iris_test_ids = np.random.permutation(len(iris_data)) #randomly splitting the data set


# In[11]:


#splitting and leaving last 15 entries for testing, rest for training
iris_train_one = iris_data[iris_test_ids[:-15]]
iris_test_one = iris_data[iris_test_ids[-15:]]
iris_train_two = iris_target[iris_test_ids[:-15]]
iris_test_two = iris_target[iris_test_ids[-15:]]


# In[12]:


iris_classify = tree.DecisionTreeClassifier()#using the decision tree for classification
iris_classify.fit(iris_train_one, iris_train_two) #training or fitting the classifier using the training set
iris_predict = iris_classify.predict(iris_test_one) #making predictions on the test dataset


# In[13]:


print(iris_predict) #labels predicted (flower species)
print (iris_test_two) #actual labels
print (accuracy_score(iris_predict, iris_test_two)*100) #accuracy metric


# In[ ]:




