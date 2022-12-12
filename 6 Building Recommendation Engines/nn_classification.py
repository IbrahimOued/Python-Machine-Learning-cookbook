# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from sklearn import neighbors, datasets

from utilities import load_data

# 2 We will use the data_nn_classifier.txt
# Load input data
input_file = 'ch06/data_nn_classifier.txt'
data = load_data(input_file)
X, y = data[:, :-1], data[:, -1].astype(np.int32)

# 3 Now, let's visualize the input data
# Plot input data
plt.figure()
plt.title('Input datapoints')
markers = '^sov<>hp'
mapper = np.array([markers[i] for i in y])
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], s=50, edgecolors='black', facecolors='none')
    # We iterate through all the datapoints and use the appropriate markers to
    # separate the classes

# 4 In order to build the classifier, we need to specify the number of nearest neighbors
# that we want to consider. Let's define this parameter

# Number of nearest neighbors to consider
num_neighbors = 10

# 5 In order to visualize the boundaries, we need to define a
# grid and evaluate the classifier on that grid. Let's define the step size
# step size of the grid
h = .01

# 6 We are now ready to build the k-nearest neighbors classifier. Let's
# define this and train it
# Create a k-neighbors classifier model and train it
classifier = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')
classifier.fit(X, y)

# 7 We need to create a mesh to plot the boundaries. Let's define this
# Create the mesh to plot the boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 8 Now let's evaluate the classifier output for all the points
# compute the outputs for all the points on the mesh
predicted_values = classifier.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

# 9 Let's plot it
# Put the computed results on the map
predicted_values = predicted_values.reshape(x_grid.shape)
plt.figure()
plt.pcolormesh(x_grid, y_grid, predicted_values, cmap=cm.Pastel1)

# 10 Now that we have plotted the color mesh, let's overlay the training
# datapoints to see where they lie in relation to the boundaries
# Overlay the training points on the map
for i in range(X.shape[0]): 
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i],  
            s=50, edgecolors='black', facecolors='none') 
 
plt.xlim(x_grid.min(), x_grid.max()) 
plt.ylim(y_grid.min(), y_grid.max()) 
plt.title('k nearest neighbors classifier boundaries')

# Now, we can consider a test datapoint and see whether the
# classifier performs correctly. Let's define it and plot it
# Test input datapoint 
test_datapoint = [4.5, 3.6] 
plt.figure() 
plt.title('Test datapoint') 
for i in range(X.shape[0]): 
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i],  
            s=50, edgecolors='black', facecolors='none')
    plt.scatter(test_datapoint[0], test_datapoint[1], marker='x',  
        linewidth=3, s=200, facecolors='black') 

    # 12 We need to extract the k-nearest neighbors classifier
    # using the following model
    # Extract k nearest neighbors 
    dist, indices = classifier.kneighbors(test_datapoint)

    # 13 Let's plot the k-nearest neighbors classifier and highlight it
    # Plot k nearest neighbors 
    plt.figure() 
    plt.title('k nearest neighbors') 
 
    for i in indices: 
        plt.scatter(X[i, 0], X[i, 1], marker='o',  
                linewidth=3, s=100, facecolors='black') 
 
        plt.scatter(test_datapoint[0], test_datapoint[1], marker='x',  
                linewidth=3, s=200, facecolors='black') 
 
        for i in range(X.shape[0]): 
            plt.scatter(X[i, 0], X[i, 1], marker=mapper[i],  
                    s=50, edgecolors='black', facecolors='none') 
        
        plt.show()

# 14 Now, let's print the classifier output on the Terminal
print("Predicted output:", classifier.predict(test_datapoint)[0])