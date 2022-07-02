#!/usr/bin/env python
# coding: utf-8

# ### CS5691 Assignment 3

# ##### Submitted by: Harshitha Nugala, Roll Number: ME19B111

# In[64]:


import numpy as np
import pandas as pd
import os
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from striprtf.striprtf import rtf_to_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm


# In[65]:


os.chdir(r"C:\Users\me19b\Downloads\PRML_assignment3 (1)\PRML_assignment3")


# In[66]:


dir_path = r"C:\Users\me19b\Downloads\PRML_assignment3 (1)\PRML_assignment3\test"
k=len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])-1
k
#not considering .DS_Store file, remove -1 if that file is not there


# ##### https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download

# In[67]:


dfo=pd.read_csv("spam.csv",encoding='latin-1',skiprows = 0)
dfo


# In[68]:


dfo.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
dfo


# In[69]:


dfn = pd.read_csv("spam_ham_dataset.csv")
dfn.drop(['Unnamed: 0', 'label_num'], axis=1, inplace=True)
dfn.rename(columns={'label': 'v1', 'text': 'v2'}, inplace=True)
dfn


# In[70]:


df=pd.concat([dfo, dfn],ignore_index=True)
df


# In[71]:


values = pd.value_counts(df["v1"])
values.plot(kind= 'bar', color= ["green", "red"])
plt.title('Spam vs Ham')


# In[72]:


values.plot(kind = 'pie',autopct='%1.0f%%')
plt.ylabel('')
plt.title('Spam vs Ham')


# In[73]:


df.isnull().sum()


# In[74]:


df.describe()


# In[75]:


y=df['v1']
y


# In[76]:


y=y.map({'spam':1,'ham':0})
y


# In[77]:


arr = []
for i in range(k):
    name = 'email' + str(i+1)
    with open(dir_path+'/'+name+'.txt') as file:
        file_contents = file.read()
        text = rtf_to_text(file_contents)
        text=[text]
    arr.append(text)


# In[78]:


df1=pd.DataFrame(arr)
df1.columns={'v2'}
df1


# In[79]:


temp = pd.concat([df['v2'], df1['v2']],ignore_index=True)
temp


# In[80]:


extraction_feature = feature_extraction.text.CountVectorizer(stop_words = 'english')
features = extraction_feature.fit_transform(temp)
features.shape


# In[81]:


X_test=features[-k:,:]
X = features[:-k,:]


# In[82]:


order = np.arange(X.shape[0])
np.random.shuffle(order)
train = order[0:int(0.75*X.shape[0])]
test = order[int(0.75*X.shape[0]):X.shape[0]]


# In[83]:


y_train = np.array(y)[train]
y_test = np.array(y)[test]


# In[84]:


x_train = X[train]
x_test = X[test]


# ### SVM - Gaussian

# In[85]:


C_list = np.arange(10, 200, 10)
score_train = np.zeros(len(C_list))
score_test = np.zeros(len(C_list))
i = 0
for c in C_list:
    print(c)
    svc = svm.SVC(C=c)
    svc.fit(x_train, y_train)
    score_test[i] = svc.score(x_test, y_test)
    score_train[i] = svc.score(x_train, y_train)
    i = i + 1 


# In[86]:


matrix = np.matrix(np.c_[C_list, score_train, score_test])
obs = pd.DataFrame(data = matrix, columns = ['C', 'Train_Accuracy', 'Test_Accuracy'])
obs


# In[87]:


best_C = obs[obs.Test_Accuracy == obs.Test_Accuracy.max()]['C']
best_C


# ##### Best C for gaussian function

# In[88]:


svc = svm.SVC(C=best_C)


# In[89]:


svc.fit(x_train, y_train)


# In[90]:


y_pred_train = svc.predict(x_train)
y_pred_test = svc.predict(x_test)
score_test = svc.score(x_test, y_test)
score_train = svc.score(x_train, y_train)


# In[91]:


score_train


# In[92]:


score_test


# In[93]:


confusion = metrics.confusion_matrix(y_test, y_pred_test)
pd.DataFrame(data = confusion, columns = ['Predicted 0', 'Predicted 1'],index = ['Actual 0', 'Actual 1'])


# ### Prediction on our test emails

# In[94]:


svc.predict(X_test)


# ### SVM - Sigmoid

# In[95]:


C_list = np.arange(10, 200, 10)
score_train = np.zeros(len(C_list))
score_test = np.zeros(len(C_list))
i = 0
for c in C_list:
    print(c)
    svc = svm.SVC(C=c, kernel='sigmoid')
    svc.fit(x_train, y_train)
    score_test[i] = svc.score(x_test, y_test)
    score_train[i] = svc.score(x_train, y_train)
    i = i + 1 


# In[96]:


matrix = np.matrix(np.c_[C_list, score_train, score_test])
obs = pd.DataFrame(data = matrix, columns = ['C', 'Train_Accuracy', 'Test_Accuracy'])
obs


# In[97]:


best_C = obs[obs.Test_Accuracy == obs.Test_Accuracy.max()]['C']
best_C


# ##### Best C for sigmoid function

# In[98]:


svc = svm.SVC(C=best_C,kernel='sigmoid')


# In[99]:


svc.fit(x_train, y_train)


# In[100]:


y_pred_train = svc.predict(x_train)
y_pred_test = svc.predict(x_test)
score_test = svc.score(x_test, y_test)
score_train = svc.score(x_train, y_train)


# In[101]:


score_train


# In[102]:


score_test


# In[103]:


confusion = metrics.confusion_matrix(y_test, y_pred_test)
pd.DataFrame(data = confusion, columns = ['Predicted 0', 'Predicted 1'],index = ['Actual 0', 'Actual 1'])


# ### Prediction on our test emails

# In[104]:


svc.predict(X_test)

