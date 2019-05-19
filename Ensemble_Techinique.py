import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth= 3)
dtf.fit(X,y)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X,y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,y)



#Method 1 : Voting

#Using Voting Method for Optimal Prediction with all three algorithms
from sklearn.ensemble import VotingClassifier
estimators = [('Decision_Tree',dtf), ('Naive_Bayes', nb), ('Log_Regr',log_reg)]
vot = VotingClassifier(estimators)
vot.fit(X,y)

dtf.score(X,y)
nb.score(X,y)
log_reg.score(X,y)

vot.score(X,y)



#Method 2 : Bagging

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(log_reg)
bag.fit(X,y)

#Random Forest a special case of Bagging
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X,y)

bag.score(X,y)
rf.score(X,y)




