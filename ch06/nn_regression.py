# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

# 2 let's generate some sample gaussian distributed data
# Generate sample data
amplitude = 10
num_points = 100
X = amplitude * np.random.rand(num_points, 1) - .5 * amplitude

# 3 We need to add some noise to the data to introduce
# some randomness into it. The goal of adding noise is to see
# wether our algorithm can get past it and still function in a robust way
# Compute target and add noise
y = np.sinc(X).ravel()
y += .2 * (.5 - np.random.rand(y.size))

# 4 Now let's visualize it as follows
# plot input data
plt.figure()
plt.scatter(X, y, s=40, c='k', facecolors='none')
plt.title('Input data')

# 5 We just generated some data and evaluated a continuous-valued function on all these points
# Let's efine a denser grid of points
# Create the 1D grid with 10 times the density of the input data
x_values = np.linspace(-.5*amplitude, .5*amplitude, 10*num_points)[:, np.newaxis]
# We define this denser grid because we want to evaluate our regressor on all of these
# points and look at how well it approximates our function

# 6 Let's now define the number of nearest neighbors that we want to consider
# Number of neighbors to consider
n_neighbors = 8

# 7 Let's initialize and train the k-nearest neighbors regressor using the paraaameters that we defined earlier
# Define and train the regressor
knn_regressor = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
y_values = knn_regressor.fit(X, y).predict(x_values)

# 8 Let's see how the regressor performs by overlapping the input and output data on top of each other
plt.figure()
plt.scatter(X, y, s=40, c='k', facecolors='none', label='input_data')
plt.plot(x_values, y_values, c='k', linestyle='--', label='predicted values')
plt.xlim(X.min() - 1, X.max() + 1) 
plt.ylim(y.min() - 0.2, y.max() + 0.2) 
plt.axis('tight') 
plt.legend() 
plt.title('K Nearest Neighbors Regressor') 
 
plt.show()
