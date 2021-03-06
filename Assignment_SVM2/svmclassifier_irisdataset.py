


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV
from prettytable import PrettyTable
from sklearn.svm import SVC
from IPython.display import HTML
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
# %matplotlib inline



# Converting the numpy dataset to pandas dataframe to allow using some pandas in-built visualisation functions
data = np.c_[df.data, df.target]                   
columns = np.append(df.feature_names, ["target"])  
irisdata = pd.DataFrame(data, columns=columns)

# Plot of various combinations of all the 4 features
sns.pairplot(irisdata,hue='target',palette='Dark2')

# The numpy dataset
irisdata.head()

# Describing the dataset
irisdata.info()

irisdata.describe()

# The feature names of the iris dataset
print(df.feature_names)

# Pre-classified target names where 0,1,2 correspond to setosa,versicolor and virginica respectively
print(df.target)

# The species of the flower to be classified
print(df.target_names)

data = df.data
target = df.target

"""<h1>Training</h1>

<h4>Taking all permuatation of the features in the dataset and kernels in SKLearn SVM Classifier and plotting their result </h4>
"""

def mesh_make(x, y, h=.02):
  x_min, x_max = x.min() - 1, x.max() + 1
  y_min, y_max = y.min() - 1, y.max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  return xx, yy

def cont_plot(ax, clf, xx, yy, **params):
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  out = ax.contourf(xx, yy, Z, **params)
  return out

"""<h1></h1>"""

s = data
t = target
result = []
kernel = ['poly','linear','rbf']
for k in kernel:
  for i in range(4):
    for j in range(i,4):
      if i!=j:
        X = s[:,[i,j]]
        y = t

        if(k=='poly'):
          model = svm.SVC(kernel=k, degree=2)
        else:
          model = svm.SVC(kernel=k)

        clf = model.fit(X, y)
        
        fig, ax = plt.subplots()
        # title for the plots
        if(k == 'poly'):
          title = ('Decision surface of SVC with kernel  '+k+" and degree = ",2)
        else:  
          title = ('Decision surface of SVC with kernel  '+k)
        # Set-up grid for plotting.
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = mesh_make(X0, X1)
        
        cont_plot(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        scatter = ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_ylabel(df.feature_names[i])
        ax.set_xlabel(df.feature_names[j])
        classes = df.target_names
        class_colours = ['#7186df','#d6e0f0','#d4625f']
        recs = []
        for f in range(0,len(class_colours)):
          recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[f]))
        plt.legend(recs,classes,loc=4)
        y_pred = model.predict(X)
        accuracy = metrics.accuracy_score(y,y_pred)
        if i != j:
          result.append([df.feature_names[i],df.feature_names[j],k,accuracy])
        print("Feature 1: ",df.feature_names[j])
        print("Feature 2: ",df.feature_names[i])
        print(df.feature_names[i],df.feature_names[j])
        print("Kernel: ",k)
        print("Accuracy: ",accuracy)
        plt.show()

"""<h1>Evaluation Metrics</h1>
<h3>Printing accuracy scores for all combinations</h3>
"""

count = 1
data = []
for row in result:
  row = [count]+row
  data.append(row)
  count = count + 1
df = pd.DataFrame(data, columns=['Serial Number','Feature 1', 'Feature 2', 'kernel','Accuracy'])
x = PrettyTable()
x.field_names = df.columns.tolist()
for row in df.values:
  x.add_row(row)
print(x)

"""<h1>Inference<h1> <h3>(Answer to Question 1 in the document)</h3>

<ul>
  <li>Looking at the table and the visualizations above, we can conclude that, when we take 2 features at a time <b>petal length</b> and <b>petal width</b> separates the data most accurately.</li>
  <li>Secondly, the corresponding kernels are polynomial kernel(Quadratic) and linear</li>
</ul>

<h1>Comparision of  OneVsAll with OneVsOne </h1>
"""

# Splitting the dataset into test and train
s_train,s_test,t_train,t_test = train_test_split(s,t,test_size=0.33,random_state=4)

#Printing shapes
print(s_train.shape)
print(s_test.shape)
print(t_train.shape)
print(t_test.shape)

# One-vs-One SVM Classifier Prediction
smodel = OneVsOneClassifier(SVC()).fit(s_train, t_train)
smodel.fit(s_train, t_train)
sprediction = smodel.predict(s_test)
print(sprediction)

# One-vs-Rest SVM Classifier Prediction
clf = OneVsRestClassifier(SVC()).fit(s_train, t_train)
spredict = clf.predict(s_test)
print(spredict)

# Actual values which should have been predicted based on testing dataset
print(t_test)

"""<h1>Evaluating the classifiers</h1>"""

# Accuracy for One-vs-One Classifier
accuracy = metrics.accuracy_score(t_test,sprediction)
print(accuracy)

# Accuracy for One-vs-Rest Classifier
accuracy1 = metrics.accuracy_score(t_test,spredict)
print(accuracy1)

# Confusion matrix
conftest = confusion_matrix(t_test, sprediction)  # One-vs-One Classifier
conf = confusion_matrix(t_test, spredict)         # One-vs-Rest Classifier
print("One Vs One Classififer")
print(conftest)
print("One Vs Rest Classifier")
print(conf)

"""<h1>Classification Report</h1>"""

from sklearn.metrics import classification_report, confusion_matrix
print("One Vs One Classififer")
print(classification_report(t_test, sprediction))  # One-vs-One Classifier
print("One Vs Rest Classifier")

print(classification_report(t_test, spredict))     # One-vs-Rest Classifier

"""<h1>Inference</h1> 
<h2>Answer to Question 2 in the document</h2>


<b>As the dataset is balanced well in our case, so both the classifiers perform equally well</b>. But in case of certain conditions like the following, one of them might be preferable <br>
<ol>
<li> One vs rest trains less no of classifier and hence is faster overall and hence is usually prefered</li>
   <li> Single classifier in one vs one uses subset of data, so single classifier is faster for one vs one</li>
   <li> One vs one is less prone to imbalance in dataset (dominance of particular classes)</li>
</ol>
<b>Easier to compute</b> : One-vs-Rest is easier to compute because it grabs a class and creates a binary label for whether a point belongs to a class or not, contrary to the One-vs-One approach in which each class is modeled against all the other classes independently.

<b>Please Note</b>
<p> We tried with a split of 0.2, we got that OneVsOne gave better results in accuracy, but that is not a fair mesaure to judge as the testing set is very small.</p>

# Training the model with all features and different kernels

**Linear Kernel**
"""

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(s_train,t_train)

t_pred = svc.predict(s_test)
print(t_pred)

cm = confusion_matrix(t_test, t_pred)
print(cm)

print("Classification Report")
print(classification_report(t_test, t_pred))

"""**Polynomial Kernel**"""

from sklearn.svm import SVC
svc = SVC(kernel = 'poly', degree=2)
svc.fit(s_train, t_train)

t_pred = svc.predict(s_test)
print(t_pred)

cm = confusion_matrix(t_test, t_pred)
print(cm)

print("Classification Report")
print(classification_report(t_test, t_pred))

"""RBF Kernel"""

from sklearn.svm import SVC
svc = SVC(kernel = 'poly', random_state = 0)
svc.fit(s_train, t_train)

t_pred = svc.predict(s_test)
print(t_pred)

cm = confusion_matrix(t_test,t_pred)
print(cm)

print("Classification Report")
print(classification_report(t_test, t_pred))

"""<h1>Finding the best hyperparameters for the model</h1>
<p>Applying GridSearch to find the best hyperparameters for the classifier. <br>The paramerters considered for grid search are as follows.</p>

<ol>
  <li>C(Regularization): This is the missclassification term. It tellls the SVM how much error can be considered.</li>
  <li>Kernel: Basically, it translates a low dimension input space and tranforms it to higher dimensions</li>
  <li>Gamma: It tells how far the influence of a single training sample reaches</li>
</ol>
"""

grid = {'C': [0.1,1, 10,], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'linear']}
gridRef = GridSearchCV(SVC(),grid,refit=True,verbose=2)
gridRef.fit(s_train,t_train)

#Best hyperparams
print(gridRef.best_estimator_)

"""Confusion Metrics"""

grid_preds = gridRef.predict(s_test)
print(confusion_matrix(t_test,grid_preds))

print(classification_report(t_test,grid_preds))

"""<h1> Grid Search Results</h1>
<h4>(Includes answer to Question 3)</h4>
<b>
<ol>
<li>Accuracy: 98% </li>
<li>Value of C : 0.1</li>
<li>Value of Gamma :  1</li>
<li>Best performing kernel : Linear</li>
</ol>
</b>
"""