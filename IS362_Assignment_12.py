#!/usr/bin/env python
# coding: utf-8

# ## IS362 Assignment 12

# In[55]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# <b>Create a pandas DataFrame with a subset of the columns in the dataset.</b>

# In[11]:


filename_mushrooms = 'agaricus-lepiota.data'
df_mushrooms = pd.read_csv(filename_mushrooms)
display(df_mushrooms.head())


# In[12]:


df_mushrooms.columns


# In[18]:


df_mushrooms.columns =['class', 'cap-shape', 'cap-syrface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing',
             'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
             'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
            'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']


# In[19]:


df_mushrooms.columns


# In[20]:


df_mushrooms


# In[27]:


df_subset_mushrooms = df_mushrooms[['class', 'odor', 'habitat']] 


# In[28]:


df_subset_mushrooms


# <b>Replace the codes used in the data with numeric values</b>

# In[41]:


df_subset_mushrooms = df_subset_mushrooms.apply(pd.to_numeric)


# In[42]:


df_subset_mushrooms


# <b>Exploratory data analysis</b>

# In[47]:


plt.figure()
pd.Series(df_subset_mushrooms['class']).value_counts().sort_index().plot(kind = 'bar')
plt.ylabel("Count")
plt.xlabel("class")
plt.title('Number of poisonous/edible mushrooms (0=edible, 1=poisonous)');


# In[50]:


odor_var = df_subset_mushrooms[['class', 'odor']]

sns.factorplot('class', col='odor', data=odor_var, kind='count', size=4.5, aspect=.8, col_wrap=4);
#plt.savefig("gillcolor1.png", format='png', dpi=500, bbox_inches='tight')


# In[51]:


habitat_var = df_subset_mushrooms[['class', 'habitat']]

sns.factorplot('class', col='habitat', data=habitat_var, kind='count', size=4.5, aspect=.8, col_wrap=4);


# In[62]:


ax1 = df_subset_mushrooms.plot.scatter(x='class',
                      y='odor',
                      c='habitat',
                          colormap='viridis')


# In[ ]:




