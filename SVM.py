import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
dataset = load_wine()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X_train,y_train)

linear_svm.score(X_test,y_test)
#88.8%
linear_svm.score(X_train,y_train)
#91.7%
linear_svm.score(X,y)
#91.01%

linear_svm = LinearSVC(C=0.1).fit(X_train,y_train)

linear_svm.score(X_test,y_test)
#86.66%
linear_svm.score(X_train,y_train)
#89.47%
linear_svm.score(X,y)
#88.76%

linear_svm = LinearSVC(C=1).fit(X_train,y_train)

linear_svm.score(X_test,y_test)
#82.22%
linear_svm.score(X_train,y_train)
#75.93%
linear_svm.score(X,y)
#77.52%

xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
Z = linear_svm.predict(np.c_[xx.ravel(),  yy.ravel()])
Z = Z.reshape(xx.shape)
pl.contour(xx, yy, Z)
pl.title('Support Vector Machine Decision Surface')
pl.axis('off')
pl.show()





