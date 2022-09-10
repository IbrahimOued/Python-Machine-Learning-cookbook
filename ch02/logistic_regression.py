# 1 Make the necessary imports let's create some sample
# data, along with training labels

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
# Here we assume that we have three classes (0, 1 and 2)
# 2 Let's initialize the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='lbfgs', C=100)

# There are a number of input parameters that can be
# specified for the preceding function, but a couple of
# important ones are solver and C.
# The solver parameter specifies the type of solver that
# the algorithm will use to solve the system of equations
# The C parameter controls the regularization strength. A
# lower value indicateds higher regularization strength

# 3 Let's train the classifier
classifier.fit(X, y)

# 4 Let's draw datapoits and boundaries. To do this, 1st we need to
# define ranges to plot the diagram, as follows:
x_min, x_max = min(X[:, 0]), max(X[:, 0]) + 1.0
y_min, y_max = min(X[:, 1]), max(X[:, 1]) + 1.0

# The precedinf values indicate the range of values that we
# Want to use in our figure. The values usually range from the
# minimum value to the maximum value present in our data.
# We add some buffers, such as 1.0, the preceding lines, for clarity

# 5 In order to plot the boundaries, we need to evaluate the function across a grid
# of points and plot it. Let's go ahead and define the grid
# denotes the step size that will be used in the mesh grid
step_size = 0.01

# define the mesh grid
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
# The x_values and y_values variables contain the grid of points where the function will be evaluated.

# 6 Let's compute the output of the classifier for all these points

# compute the classifier output
mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

# reshape the array
mesh_output = mesh_output.reshape(x_values.shape)

# 7 Let's plot the boundaries using colored regions:
# plot the ouput using a colored plot
plt.figure()

# choose a color scheme you can find all the options
# here http://matplotlib.org/examples/color/colormaps_reference.html
plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
# This is basically a 3D plotter that takes the 2D points and the associated values to draw different regions using a color scheme. 

# 8 Let's overlay the training points on the plot
# Overlay the training points on the plot 
plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

# specify the boundaries of the figure
plt.xlim(x_values.min(), x_values.max())
plt.ylim(y_values.min(), y_values.max())

# specify the ticks on the X and Y axes
plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

plt.show()

# Here, plt.scatter plots the points on the 2D graph. X[:, 0] 
# specifies that we should take all the values along the 0 axis 
# (the x axis in our case), and X[:, 1] specifies axis 1 
# (the y axis). The c=y parameter indicates the color sequence.
# We use the target labels to map to colors using cmap. Basically, 
# we want different colors that are based on the target labels. 
# Hence, we use y as the mapping. The limits of the display 
# figure are set using plt.xlim and plt.ylim. In order to mark the 
# axes with values, we need to use plt.xticks and plt.yticks.
# These functions mark the axes with values so that it's easier 
# for us to see where the points are located. In the preceding 
# code, we want the ticks to lie between the minimum and maximum 
# values with a buffer of one unit. Also, we want these ticks 
# to be integers. So, we use the int() function to round off the values.

# 9 Run the code

# 10 Let's see how the c parameter affects our model? The c parameter indicates
# the penalty for misclassification. If we set it to 1.0, we will get the another plot

# 11 And another by setting it to 1000
# As we increase c, there is a higher penalty for misclassification. Hence, the boundaries become more
# optimized