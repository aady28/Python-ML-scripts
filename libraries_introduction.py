a = 10
b = [10, "rahul", 23.45]
c = {"a" : 12, "b" : 15, "c" : 20}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


u = np.array([2, 5])
v = np.array([3, 1])

def plot_vector2d(vector2d, origin = [0, 0], **options):
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1], **options)

    

plot_vector2d(u, color = "r") 
plot_vector2d(v, color = "b")   
plot_vector2d(v, origin = u, color = "b")
plot_vector2d(u, origin = v, color = "r")
plot_vector2d(u+v, color = "g")
plt.axis([0, 9, 0, 7])
plt.text(0.7, 3, "u", color = "r", fontsize = 18)
plt.text(2, 0.3, "v", color = "b", fontsize = 18)
plt.text(3, 5.6, "v", color = "b", fontsize = 18)
plt.text(4, 2.7, "u", color = "r", fontsize = 18)
plt.text(2.5, 3, "u+v", color = "g", fontsize = 18)
plt.grid()
plt.show()


################################# NUMPY ###########################################

import numpy as np
my_arr = np.arange(1000000)
my_list = list(range(1000000))

print("Time elapsed by Numpy array : ")
%time for a in range(10): my_arr2 = my_arr * 2
print("Time elapsed by Python list : ")
%time for a in range(10): my_list2 = my_list * 2

import time
    
for a in range(10):
    start = time.time()
    my_arr2 = my_arr * 2
    end = time.time()
    
print(end - start)

data = np.random.randn(2, 3)
data

data * 10
data + data

data.shape
data.dtype

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

arr1

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)

arr2
arr2.ndim
arr2.shape

arr1.dtype
arr2.dtype

np.zeros(10)
np.zeros((2, 3))
np.empty((2, 3, 2))
np.arange(10, step = 0.01)



arr1 = np.array([1, 2, 3], dtype = np.float64)
arr1.dtype

arr2 = np.array([1, 2, 3], dtype = np.int32)
arr2.dtype

arr = np.array([1, 2, 3, 4, 5])
arr.dtype

float_arr = arr.astype(np.float64)
float_arr.dtype

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr.dtype

arr.astype(np.int32)

numeric_strings = np.array(['1.25', '-9.6', '42'], dtype = np.string_)
numeric_strings.dtype

numeric_strings.astype(np.float64)
numeric_strings.dtype
numeric_strings

int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50])

int_array.astype(calibers.dtype)

empty_uint32 = np.empty(8, dtype = 'u4')
empty_uint32

arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr

arr * arr
arr - arr

1 / arr

arr ** 0.5

arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr2
arr

arr2 > arr

arr = np.arange(10)
arr

arr[5]

arr[5:8]

arr[5:8] = 12

arr

z = [0, 1, 2, 3, 4, 5]
z[2:4] = list([10, 10])
z

arr_slice = arr[5:8]
arr_slice
arr_slice[1] = 12345
arr
arr_slice[:] = 64
arr

test = arr[5:8].copy()
del(test)

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr2d[1]
arr2d[1, 1]
arr2d[1][1]

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d.shape
arr3d[0]
arr3d[1]

old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d

arr3d[1, 0]
x = arr3d[1]
x
x[0]
arr
arr[1:6]


arr2d
arr2d[:2, 1:]
arr2d[1, :2]
arr2d[:2, 2]
arr2d[:, :1]
arr2d[:2, 1:] = 0
arr2d

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)

names
data

names == "Bob"
data[names == "Bob"]
data[names == "Bob", 2:]
data[names == "Bob", 3]
names != "Bob"
data[~(names == "Bob")]
cond = names == "Bob"
data[~cond]

mask = (names == "Bob") | (names == "Will")
test_data = data[mask]
test_data
test_data = 0
data

data[data <0] = 0
data
data[names != "Joe"] = 7
data

arr = np.empty((8, 4))
arr

for i in range(8):
    arr[i] = i

arr[[4, 3, 0, 6]]
arr[[-3, -5, -7]]

arr = np.arange(32).reshape((8, 4))
arr

arr[[1, 5, 7, 2], [0, 3, 1, 2]]

arr[[1, 7]][:, [3, 2, 1, 0]]

# Boolean comparisons and Fancy Indexing always result in new arrays

arr = np.arange(15).reshape((3, 5))
arr

arr.T

arr = np.random.randn(6, 3)
arr

np.dot(arr.T, arr)

arr = np.arange(16).reshape(2, 2, 4)
arr
arr.T
arr.transpose((2, 0, 1))
arr.strides
arr
arr.swapaxes(1, 2)

arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)

x = np.random.randn(8)
y = np.random.randn(8)
x
y
np.maximum(x, y)

arr = np.random.randn(7) * 5
remainder, whole_part = np.modf(arr)
remainder
whole_part

arr
np.sqrt(arr)
np.sqrt(arr, arr)
arr

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
xs
ys

z = np.sqrt(xs ** 2 + ys ** 2)
z

import matplotlib.pyplot as plt
plt.imshow(z, cmap = plt.cm.gray)
plt.colorbar()
plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

cond = np.array([True, False, True, True, False])

result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
result

result = np.where(cond, xarr, yarr)
result

arr = np.random.randn(4, 4)
arr

arr > 0

np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)

arr = np.random.randn(5, 4)
arr
arr.mean()
arr.sum()
np.mean(arr)
np.sum(arr)
arr.mean(axis = 0)
arr.sum(axis = 0)

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr
arr.cumsum()

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr

arr.cumsum(axis = 0)
arr.cumsum(axis = 1)

arr.cumprod(axis = 0)
arr.cumprod(axis = 1)

arr = np.random.randn(100)
arr

(arr > 0).sum()

bools = np.array([False, False, True, False])
bools.any()
bools.all()

arr = np.random.rand(6)
arr
arr.sort()
arr

arr = np.random.randn(5, 3)
arr.sort(axis = 0)
arr.sort(axis = 1)
arr

arr[4].sort()
arr

large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)

ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)

values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')
np.savez('array_archive.npz', a = arr, b = arr)
arch = np.load('array_archive.npz')
arch['a']

np.savez_compressed('arrays_compressed.npz', a = arr, b = arr)

####################################### Linear Algebra #############################

x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])

x
y

x.dot(y)

np.dot(x, np.ones(3))
x @ np.ones(3)

from numpy.linalg import inv, qr
X = np.random.randn(5, 5)
mat = X.T.dot(X)
inv(mat)

mat.dot(inv(mat))

q, r = qr(mat)
q
r

########################################## Pseudorandom Random Generation #################

samples = np.random.normal(size = (4, 4))
samples

from random import normalvariate
N = 1000000

%timeit np.random.normal(size = N)
%timeit samples = [normalvariate(0, 1) for _ in range(N)]
np.random.seed(1234)

rng = np.random.RandomState(1234)
rng.randn(10)

import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

plt.plot(walk[:100])    

nsteps =  1000
draws = np.random.randint(0, 2, size = nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
walk.min()
walk.max()
(np.abs(walk) >= 10).argmax()

nwalks = 5000
nsteps = 1000

draws = np.random.randint(0, 2, size = (nwalks, nsteps))
draws

steps = np.where(draws > 0, 1, -1)
steps

walks = steps.cumsum(1)
walks
walks.min()
walks.max()

hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum()

crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()

test = np.abs(walks[hits30])

####################################### Pandas ######################################

import pandas as pd
from pandas import Series, DataFrame

obj = pd.Series([4, 7, -5, 3])
obj
obj.values
obj.index

obj2 = pd.Series([4, 7, -5, 3], index = ['d', 'b', 'a', 'c'])
obj2
obj2.values
obj2.index

obj2['a']
obj2['d']
obj2[['c', 'a', 'd']]

obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)

'b' in obj2
'e' in obj2

sdata = {'Ohio' : 35000, 'Texas' : 71000, 'Oregon' : 16000, 'Utah' : 5000}
obj3 = pd.Series(sdata)
obj3

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index = states)
obj4

pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()

obj3
obj4

obj3 + obj4
obj4.name = 'population'
obj4.index.name = 'state'
obj4
obj
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']

data = {'state' : ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year' : [2000, 2001, 2002, 2001, 2002, 2003],
        'pop' : [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]
        }
data
frame = pd.DataFrame(data)
frame.head()
pd.DataFrame(frame, columns = ['year', 'state', 'pop'])

frame2 = pd.DataFrame(data, columns = ['year', 'state', 'pop', 'debt'], index = ['one', 'two', 'three', 'four', 'five', 'six'])
frame2
frame2.columns
frame2['state']
frame2.year

frame2.loc['five']
frame2['debt'] = 16.5
frame2.debt = np.arange(6.)
frame2

val = pd.Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])
frame2['debt'] = val

frame2['eastern'] = frame2.state == "Ohio"
del frame2['eastern']
frame2.columns

pop = {'Nevada' : {2001 : 2.4, 2002 : 2.9},
       'Ohio' : {2000 : 1.5, 2001 : 1.7, 2002 : 3.6}
       }

frame3 = pd.DataFrame(pop)
frame3
frame3.T
# pd.DataFrame(pop, index = [2001, 2002, 2003])
frame3['Ohio'][:-1]
frame3['Nevada'][:2]

pdata = {'Ohio' : frame3['Ohio'][:-1],
         'Nevada' : frame3['Nevada'][:2]
         }
pdata
pd.DataFrame(pdata)

frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3.values
frame2.values

obj = pd.Series(range(3), index = ['a', 'b', 'c'])
obj
index = obj.index
index[1:]
index[1]

label = pd.Index(np.arange(3))
label

obj2 = pd.Series([1.5, -2.5, 0], index = label)
obj2
obj2.index is label

frame3
frame3.columns

'Ohio' in frame3.columns
2003 in frame3.index

dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
dup_labels

obj = pd.Series([4.5, 7.2, -5.3, 3.6], index = ['d', 'b', 'a', 'c'])
obj

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2

obj3 = pd.Series(['blue', 'purple', 'yellow'], index = [0, 2, 4])
obj3

obj3.reindex(range(6), method = 'ffill')

frame = pd.DataFrame(np.arange(9).reshape((3, 3)), index = ['a', 'c', 'd'], columns = ['Ohio', 'Texas', 'California'])
frame
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2

states = ['Texas', 'Utah', 'California']
frame.reindex(columns = states)
frame.loc[['a', 'b', 'c', 'd'], states]

obj = pd.Series(np.arange(5.), index = ['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])

data = pd.DataFrame(np.arange(16).reshape((4, 4)), 
                    index = ['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns = ['one', 'two', 'three', 'four']
                    )
data
data.drop(['Colorado', 'Ohio'])
data.drop('two', axis = 1)
data.drop(['two', 'four'], axis = 'columns')
obj.drop('c', inplace = True)
obj



#######################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

m = np.array([[2, 3], [4, 5]])
m1 = np.array([[2, 3], [4, 5]])

m * m1
m @ m1

x = np.arange(-10, 10, 0.01)
y = (1 / (1 + np.power(np.e, -x)))

plt.plot(x, y)
plt.show()

my_list = list(range(1000000))
my_arr = np.array(range(1000000))

%time for i in range(10): my_list2 = my_list * 2
%time for i in range(10): my_arr2 = my_arr * 2







    

    
    
    













































































































































































































































































































































































































































































































































































    






















































































































































































































































































































































    









































































    