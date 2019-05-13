#!/usr/bin/env python
# coding: utf-8

# In[3]:


filen = "C:\\Users\\Maruthi\\Downloads\\creditcard.csv"


# In[1]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))
print("Sklearn : {}".format(sklearn.__version__))


# In[4]:


# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data = pd.read_csv(filen)


# In[6]:


data.head()


# In[7]:


print(data.columns)


# In[8]:


print(data.shape)


# In[9]:


print(data.describe())


# In[10]:


data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)


# In[11]:


# plot histogram of each parameter

data.hist(figsize = (20,20))
plt.show()


# In[12]:


# checking number of frauds in dataset

Fraud = data[data["Class"] ==1]
Valid = data[data["Class"] ==0]

outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)
print("Fraud Cases: {}".format(len(Fraud)))
print("Valid Cases: {}".format(len(Valid)))


# In[13]:


# corr matrix witrh heatmap
# we see most of the values are 0, i.e no correlation between v variables
# but, we care about the relation between class and v variables
# we see the variations
# lighter ones will be a positive correlation 
# darker ones will be a negative correlation 

corrmat = data.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat , vmax = 8, square = True)
plt.show()


# In[14]:


x = data.drop("Class" , axis = 1)

target = "Class"
y = data[target]

# here we have x with 30 columns, class column has been dropped
# and y a single array with target values
print(x.shape)
print(y.shape)


# In[15]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# In[16]:


# applying algorithms
# how successful are we in outlier detection
# anomoly detection
# we can also use svm for outlier detection but because of huge dataset, it might take longer time
# hence we use IsolationForest & LocalOutlierFactor


# In[17]:



# LocalOutlierFactor is an is an unsupervised outlier detection method
# it measures the local anomaly score of a sample with its respective neighbours
# the score depends on the fact how isolated a sample is from its neighborhood
# isolationforest, returns anomaly score of each sample, it isolates the observations
# by randomly selecting a feature
# and then by randomly selecting a split between the max and min value of the selected feature 
# here, we will compare results of IsolationForest & LocalOutlierFactor


# In[18]:



# define a random state
state = 1


# In[19]:


# define the outlier detection method
# in contamination we will pass outlier fraction, i.e the number of outliers we think to be present in the data
# higher the percentage of outliers in the dataset, higher the amount of n_neighbors we want

classifiers = {
    "Isolation Forest": IsolationForest(max_samples = len(x) , contamination = outlier_fraction , random_state=state) ,
    
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20 , contamination=outlier_fraction) 
}


# In[20]:


# fit the model
n_outliers = len(Fraud)

for i , (clf_name , clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(x)
        scores_pred = clf.negative_outlier_factor_
        
    else:
        clf.fit(x)
        scores_pred = clf.decision_function(x)
        y_pred = clf.predict(x)

   

 # the results in y prediction to give us a -1 for a outlier and 1 for a in liar 
 # we can take our all in liars and classify them as 0 or valid transaction 
 # we can take our all outliers, the ones we think do not belong to the rest, can be classified as 1 or fraud transaction   
    
    
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != y).sum()
    
    print("{} : {}".format(clf_name , n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))


# In[21]:


# we got 99 percent accuracy, because we had lot of valid cases
# and 30 percent precision in predicting fraudlent transactions using Isolation Forest

