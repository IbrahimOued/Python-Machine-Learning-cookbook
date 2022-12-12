# Let's import the necessary librairies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import  GaussianNB

# 2 You were provided with a data_mutlivar.txt file. This
# contains data that will be use here. This contains
# comma-separated numerical data in each line. Let's load the data
# from this file

input_file = 'ch02/data_multivar.txt'
X, y = [], []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])
X = np.array(X)
y = np.array(y)

# We have now loaded the input data into x the labels into y.
# There are 2 labels: 0, 1, 2 and 3

# 3 Let's build the Naive Bayes classifier
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)

# The gaussiannb function specifies the Gaussian Naive Bayes model

# 4 Let's compute the accuracy measure of the classifier
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")
# The following accuracy is returned: 99.5

# 5 Let's plot the data and the boundaries. We will use the procedure
# followed in the previous recipe, Building a logistic regression classifier:
x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

# denotes the step size that will be used in the mesh grid
step_size = 0.01

# define the mesh grid
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# compute the classifier output
mesh_output = classifier_gaussiannb.predict(np.c_[x_values.ravel(), y_values.ravel()])

# reshape the array
mesh_output = mesh_output.reshape(x_values.shape)

# Plot the output using a colored plot 
plt.figure()

# choose a color scheme 
plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

# Overlay the training points on the plot 
plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

# specify the boundaries of the figure
plt.xlim(x_values.min(), x_values.max())
plt.ylim(y_values.min(), y_values.max())

# specify the ticks on the X and Y axes
plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

plt.show()

# There is no restriction on the boundaries to be linear here.
# In the preceding recipe, Building a logistic regression classifier,
# we used up all the data for training. A good practice in machine
# learning is to have non-overlapping data for training and testing.
# Ideally, we need some unused data for testing so that we can get an
# accurate estimate of how the model performs on unknown data.
# There is a provision in scikit-learn that handles this very well, as shown in the next recipe.