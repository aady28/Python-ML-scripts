import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from sklearn.datasets import load_iris
dataset = load_iris()


X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

#DT alogo
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 3)
dtf.fit(X_train, y_train)


dtf.score(X_test, y_test)
dtf.score(X_train, y_train)
dtf.score(X, y)

dtf.predict([[5.8,3.2,1,2.4]])

from sklearn import tree
iris_tree = tree.export_graphviz(dtf, out_file = None)
graph = graphviz.Source(iris_tree)
graph.render("irisTree")

irisTree = tree.export_graphviz(dtf, out_file=None, 
                     feature_names=dataset.feature_names,  
                     class_names=dataset.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(irisTree)  
graph.render("irisTreeCol") 



#KNN algo
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knn.score(X_test, y_test)
knn.score(X_train, y_train)
knn.score(X, y)


#from sklearn.gaussian_process import 



