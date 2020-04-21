#!/usr/bin/env python
# coding: utf-8

# In[84]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# In[85]:

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import classification_report , accuracy_score
from pandas.plotting import scatter_matrix

# In[86]:

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id','clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

df = pd.read_csv(url, names=names)

# In[87]:


df.replace('?' ,-99999 , inplace =True)
print(df.axes)
df.drop('id' , 1, inplace=True)
print(df.shape)

# In[89]:

df.describe()
df.head()
print(df.loc[99])

# In[37]:

df.hist(figsize = (8,8))
plt.show()

# In[94]:

scatter_matrix(df , figsize = (20,20))
plt.show()

# In[96]:

y = np.array(df['class'])
X = np.array(df.drop(['class'] , 1))

from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X, y , test_size = 0.2)

models = []
models.append(('KNN' , KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM' , SVC()))

for name , model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test , predictions))
    print(classification_report(y_test , predictions))






