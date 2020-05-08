#!/usr/bin/env python
# coding: utf-8

# # Impoting Required Libraries (The actual algorithm is self coded)

# In[46]:


import pandas as pd 
import random
import numpy as np
import pprint 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
get_ipython().run_line_magic('matplotlib', 'inline')
# to generate your own dataset uncomment this code
# data = {'Feature1':[random.uniform(0, 6) for i in range(100)], 'Feature2':[random.uniform(0, 6) for i in range(100)], 'Feature3':[random.uniform(0, 6) for i in range(100)], 'Feature4':[random.uniform(0,6) for i in range(100)],'Output':[round(random.random()) for i in range(100)]} 
# df = pd.DataFrame(data) 


# # Load Dataset

# In[8]:


#load Dataset
df = pd.read_csv("data_d.csv")
df.head()


# # Exploratory Data Analysis and Visualisation

# In[9]:


df.describe()


# In[10]:


df.shape


# In[11]:


df.info()


# 

# In[12]:


#countplot
ax = sns.countplot(df["diagnosis"],label="Count")      


# In[28]:


#Correlation matrix
corr = df.corr(method = 'pearson')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 275, as_cmap=True)

sns.heatmap(corr, cmap=cmap, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)


# # Data Preprocessing

# In[14]:


#preprocessing
df = df.drop('id',axis=1)
df = df.drop('Unnamed: 32',axis=1)
df.columns.shape
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})


# In[15]:


df.head()


# # Splitting Dataset to train test

# In[44]:


d = df['diagnosis'].values
X = df.drop('diagnosis',axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X,d, test_size=0.30)


# # Single Layer Perceptron Model

# In[55]:


#perceptron
finals = []
from sklearn.metrics import accuracy_score
import random
class Perceptron:
    def __init__(self,input_size,epochs=100,alpha=0.02):
        self.epochs = epochs
        self.alpha = alpha
#         self.input_size = input_size
        weight = [random.random() for i in range(input_size+1)]
        weight[0] = 1.0
        self.weight = weight
        print("initial weight is ")
        print(weight)
#         self.weight = np.zeros(input_size+1)
    def activation(self,x):
        return 1 if x>=0 else 0
    def predict(self,x):
#         print(self.weight.T)
#         print(X)
       
        z = np.dot(self.weight,x)
#         print(z)
#         print(self.activation(z))
        a = self.activation(z)
        return a
   
    
    def learn(self,X,d):
#         final = []
        for j in range(self.epochs):
            sum = 0
            for i in range(d.shape[0]):
                x = np.insert(X[i],0,1)
                y = self.predict(x)
                e = (d[i] - y)
                self.weight = self.weight + self.alpha*e*x
                sum = sum + e
            finals.append(abs(sum))
            print("epoch = "+str(j)+"   error = "+str(abs(sum)))
#             print("Error" +str(sum))
#             print(self.weight)
        print("learning curve")
        plt.plot(range(self.epochs),finals)
                
if __name__ == '__main__':

    

    input_size = 30
    learning_rate = 0.02
    iterations = 30
    perceptron = Perceptron(input_size=30,alpha = learning_rate,epochs = iterations)
    perceptron.learn(X_train,y_train )


    print("The learned weight is ")
    print(perceptron.weight)

print("Learning Rate = "+str(learning_rate))
y_pred_train = []
for i in range(X_train.shape[0]):
    x= np.insert(X_train[i],0,1)
    y_pred_train.append(perceptron.predict(x))
  
    
print("Training data Accuracy = " +str(accuracy_score(y_train, y_pred_train)))
conftrain = confusion_matrix(y_train,y_pred_train)

y_pred_test = []
finals2 = []

for i in range(X_test.shape[0]):
    x= np.insert(X_test[i],0,1)
    y_pred_test.append(perceptron.predict(x))
#     print(y_pred_test)
# print("Hi");print(X_train)
# print(X_train)
# print(y_train)
# print(y_pred_train)

# print(y_pred)
print("Testing data Accuracy = " +str(accuracy_score(y_test, y_pred_test)))
conftest = confusion_matrix(y_test, y_pred_test)


# # Evaluation Metrics

# In[56]:


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
print("Confusion Matrix:Testing Data")
df_cm = pd.DataFrame(conftest, index = [i for i in ("True","False")],
                  columns = [i for i in ("True","False")])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

print("Confusion Matrix:Training Data")
df_cm = pd.DataFrame(conftrain, index = [i for i in ("True","False")],
                  columns = [i for i in ("True","False")])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# ### Training Precision and Recall

# In[59]:



precision = precision_score(y_train,y_pred_train)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_train,y_pred_train)
print('Recall: %f' % recall)


# ### Testing Precision and Recall

# In[72]:



precision = precision_score(y_test,y_pred_test)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test,y_pred_test)
print('Recall: %f' % recall)


# # Getting Top 5 Features

# In[43]:


from sklearn.feature_selection import SelectKBest, chi2
X_new = SelectKBest(chi2, k=3).fit_transform(X,d)
X_new.shape
features_columns = df.columns

fs = SelectKBest(k=5)
fs.fit(X,d)
top_features = zip(fs.get_support(),features_columns)

print("The top 5 features are: ")
pp = pprint.PrettyPrinter(depth=4)
for i,j in top_features:
    if i==True:
        pp.pprint(j)


# # Plot of Top 2 Features with the target value

# In[45]:


plt.scatter(df[ df['diagnosis']==0.0 ]['texture_mean'], df[ df['diagnosis']==0.0 ]['concavity_mean'], marker='o', Label=0)
plt.scatter(df[ df['diagnosis']==1.0 ]['texture_mean'], df[ df['diagnosis']==1.0 ]['concavity_mean'], marker='x', Label=1)
plt.xlabel('texture_mean')
plt.ylabel('concavity_mean')
plt.show()


# # Final Remarks:
# 
#     1. Training data Accuracy = 0.9296482412060302
#     2. Testing data Accuracy = 0.8830409356725146
#     3. Top 5 Features: 
# 
#         *texture_mean
#         *concavity_mean
#         *fractal_dimension_se
#         *texture_worst
#         *concavity_worst
# 
# * With the Single layer perceptron algorithm, we got the above results, the results can be improved using a MultiLayerPerceptron.
# * The accuracy obtained using our code is almost in sync with any of the standard libraries like SKlearn, Tensorflow etc.

# # Paper Referred

# Link: https://ieeexplore.ieee.org/abstract/document/8819775