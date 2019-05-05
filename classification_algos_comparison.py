#Classification Algorithms comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)



#Naiive Bayes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

gnb.score(X_test,y_test)       #94.73%
gnb.score(X_train,y_train)     #96.42%
gnb.score(X,y)                 # 96%




#KNN Algorithm

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knn.score(X_train, y_train)     #98.21%
knn.score(X_test, y_test)       #97.36%
knn.score(X, y)                 #98%



#Decision Tree

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 3)
dtf.fit(X_train, y_train)


dtf.score(X_test, y_test)       #92.1%
dtf.score(X_train, y_train)     #99.1%
dtf.score(X, y)                 #97.33%





