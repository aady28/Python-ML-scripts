import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('dataset/AirQualityUCI.xlsx')

X = dataset.iloc[:, 2:12].values
y1 = dataset.iloc[:, -3].values
y2 = dataset.iloc[:, -2].values
y3 = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imp = Imputer()
X = imp.fit_transform(X)

#X_c = np.c_[np.ones(100), X]

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y1)
lin_reg.score(X, y1)

lin_reg.fit(X, y2)
lin_reg.score(X, y2)

lin_reg.fit(X, y3)
lin_reg.score(X, y3)

# Maximum accuracy with y3 - AH
#__________________________________________

GT = X[:,7]
NO2 = X[:,8]
O3 = X[:,9]

GT = GT.reshape(-1, 1)
NO2 = NO2.reshape(-1, 1)
O3 = O3.reshape(-1, 1)

lin_reg1 = LinearRegression()

#Testing which gas is affecting humidity (y1) the most
lin_reg1.fit(GT, y1)
lin_reg1.score(GT, y1)

lin_reg1.fit(NO2, y1)
lin_reg1.score(NO2, y1)

lin_reg1.fit(O3, y1)
lin_reg1.score(O3, y1)

# Hence NO2 affects the most



#Testing which gas is affecting humidity (y2) the most
lin_reg1.fit(GT, y2)
lin_reg1.score(GT, y2)

lin_reg1.fit(NO2, y2)
lin_reg1.score(NO2, y2)

lin_reg1.fit(O3, y2)
lin_reg1.score(O3, y2)

# Hence NO2 affects the most



#Testing which gas is affecting humidity (y1) the most
lin_reg1.fit(GT, y3)
lin_reg1.score(GT, y3)

lin_reg1.fit(NO2, y3)
lin_reg1.score(NO2, y3)

lin_reg1.fit(O3, y3)
lin_reg1.score(O3, y3)

# Hence NO2 affects the most



