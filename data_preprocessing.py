import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.random.randint(0,100)
y = 4 + 5*X + np.random.randint(0,100)

#fill.na for outliers
#mat = np.array([1,2],[3,4])
dataset
X = dataset.iloc


#my_list = list(range(1000000))
#my_arr = np.array(range(1000000))

#%time for i in range(10): my_list2 = my_list * 2
#%time for i in range(10): my_arr2 = my_arr * 2


plt.scatter([1, 2, 3], [4, 5, 6])
plt.show()


#plt.plot([1, 2, 3], [4, 5, 6])
#plt.show()

#a = pd.Series([1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e'])
#
#b = pd.DataFrame({1 : [1, 2, 3, 4, 5],
#                  2 : [1, 2, 3, 5, 5]})
#
#c = pd.DataFrame([[1, 2, 3, 4, 5],
#                  [1, 2, 3, 4, 5],
#                  [1, 2, 3, 4, 5, 6]])


#dataset = pd.read_csv('dataset/housing.csv') 

#plt.scatter(dataset['total_bedrooms'], dataset['total_rooms'])
#plt.show()

#pd.scatter_matrix(dataset)

#dataset = pd.read_csv('dataset/Data_Pre.csv')


#X = dataset.iloc[:, 0:3].values
#y = dataset.iloc[:, -1].values

#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imp.fit(X[:, 0:2])
#X[:, 0:2] = imp.transform(X[:, 0:2])
#
#from sklearn.preprocessing import LabelEncoder
#lab = LabelEncoder()
#X[:, 2] = lab.fit_transform(X[:, 2])
#y = lab.fit_transform(y)
#lab.classes_
#
#from sklearn.preprocessing import OneHotEncoder
#one = OneHotEncoder(categorical_features = [2])
#X = one.fit_transform(X)
#X = X.toarray()
#
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)

dataset = pd.read_csv('Datasets/adult.csv', names = ['age', 'workclass','fnlwgt','education',
                                                     'education-num','marital-status',
                                                     'occupation','relationship',
                                                     'race','sex','capital-gain'
                                                     ,'capital-loss','hours-per-week',
                                                     'native-country','salary'])


print(dataset.describe())

from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imp1 = Imputer(missing_values = '?', strategy = 'mean', axis = 0)
imp2 = Imputer(missing_values = '?', strategy = 'mode', axis = 0)
imp3 = Imputer(missing_values = '?', strategy = 'median', axis = 0)

pd.scatter_matrix(dataset)

#X = dataset.iloc[:, 0:1].values
#imp.fit(X[:, 0:1])
#X[:, 0:1] = imp.transform(X[:, 0:1])
#

imp1.fit(dataset.iloc[:, 10:12])
dataset[:, 0:2] = imp1.transform(dataset[:, 10:12])

imp2.fit(dataset.iloc[:, 10:12])
dataset[:, 0:2] = imp2.transform(dataset[:, 10:12])

imp3.fit(dataset.iloc[:, 0].values)
dataset[:, 0] = imp3.transform(dataset[:, 0])


#from sklearn.preprocessing import LabelEncoder
#lab = LabelEncoder()
#X[:, 2] = lab.fit_transform(X[:, 2])
#y = lab.fit_transform(y)
#lab.classes_
#
#from sklearn.preprocessing import OneHotEncoder
#one = OneHotEncoder(categorical_features = [2])
#X = one.fit_transform(X)
#X = X.toarray()

 #To transform / standardize data
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)


