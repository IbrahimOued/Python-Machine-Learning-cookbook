# 1 Let's make the usual imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 2 Let's create some sample 2-dimentional data
# Input data
X = np.array([[1, 1], [1, 3], [2, 2], [2.5, 5], [3, 1],
              [4, 2], [2, 3.5], [3, 3], [3.5, 4]])

# 3 Our goal is to find the three closest neighbors to
# any given point, so let's define this parameter:
# Number of neighbors we want to find
num_neighbors = 3

# 4 Let's define a random datapoint that's not present in the input data:
# Input point
input_point = [[2.6, 1.7]]

# 5 We need to see what this data looks like; let's plot it, as follows:
# Plot datapoints
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, color='k')

# 6 In order to find the nearest neighbors, we need to define the
# NearestNeighbors object with the right parameters and train it on the input data:
# Build nearest neighbors model
knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)

# 7 We can now find the distances parameter of the input point to all the points in the input data:
distances, indices = knn.kneighbors(input_point)

# 8 We can print k nearest neighbors, as follows
# Print the 'k' nearest neighbors
print("k nearest neighbors")
for rank, index in enumerate(indices[0][:num_neighbors]):
    print(str(rank+1) + " -->", X[index])
    # The indices array is already sorted, so we just need
    # to parse it and print the datapoints.

# 9 Now, let's plot the input datapoint and highlight the k-nearest neighbors:
# Plot the nearest neighbors
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, color='k')
plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1],
            marker='o', s=150, color='k', facecolors='none')
plt.scatter(input_point[0][0], input_point[0][1],
            marker='x', s=150, color='k', facecolors='none')

plt.show()
