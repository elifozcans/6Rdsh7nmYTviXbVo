# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:42:53 2022

@author: ELİF / PROJECT
"""

#required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#import dataset

dataset = pd.read_csv('C:/Users/EMRE/Desktop/term-deposit-marketing-2020.csv')
print(dataset)

#Data Visualization and Analysis
dataset.shape
dataset['y'].value_counts() #no:37104 #yes:2896
dataset.columns
dataset.values
dataset.describe(include='all')

X = dataset.iloc[:,[0,5,9,11,12]]
X.head()
y = dataset.iloc[:,-1]
y.head()

#Data normalization
X = preprocessing.StandardScaler().fit_transform(X)
X[0:4]

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)
y_test.shape

#Training and Predicting 
knnmodel=KNeighborsClassifier(n_neighbors=3)
knnmodel.fit(X_train,y_train)
classifier = KNeighborsClassifier(algorithm='auto', metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     weights='uniform')
y_predict1=knnmodel.predict(X_test)

#Accuracy
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_predict1)
acc #0.92

#Output Visualization
prediction_output=pd.DataFrame(data=[y_test.values,y_predict1],index=['y_test','y_predict1'])
prediction_output.transpose()
prediction_output.iloc[0,:].value_counts() #no:11150 #yes:850

#Confusion Matrix  #939:yanlış sınıflandırma #11061:doğru sınıflandırma
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values,y_predict1)
cm

#En iyi k değerini bulalım
Ks=10
mean_acc=np.zeros((Ks-1))

#train and predict
for n in range(1,Ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1]=accuracy_score(y_test,yhat)
    
print(mean_acc)    
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) #The best accuracy was with 0.9304166666666667 with k= 8

plt.plot(range(1,Ks),mean_acc,'g')
plt.legend(('Accuracy '))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

#5-fold cross validation (Success Metric)
from sklearn.model_selection import cross_val_score
# use the same model as before
knn = KNeighborsClassifier(n_neighbors = 5)

# X,y will automatically devided by 5 folder, the scoring I will still use the accuracy
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy') 

# print all 5 times scores 
print(scores) #[0.9255   0.919125 0.916375 0.90875  0.91675 ]

# then I will do the average about these five scores to get more accuracy score.
print(scores.mean()) #0.9173

