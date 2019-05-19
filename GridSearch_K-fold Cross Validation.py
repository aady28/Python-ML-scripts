import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

#Now we specify the paramters and their values based on which we need to form grid to find optimal value
param_grid = [{'criterion': ['gini', 'entropy']},
                {'max_depth' : [3,4,5,6,7,8,9]}]


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(dtf, param_grid)
grid.fit(X,y)

grid.best_estimator_
grid.best_params_
grid.best_score_


