# 1 Let's make the basic imports
from audioop import mul
import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt

# 2 Let's define parameters to generate some training data
# Generate training data
min_value = -12
max_value = 12
num_datapoints = 90

# 3 This training data will consist of a function we define that will transform the
# values. We expect the neural network to learn this on its own, based on the input values
# that we provide
x = np.linspace(min_value, max_value, num_datapoints) 
y = 2 * np.square(x) + 7 
y /= np.linalg.norm(y) 

# 4 Reshape the arrays
data = x.reshape(num_datapoints, 1) 
labels = y.reshape(num_datapoints, 1)

# 5 Plot the input data:
plt.figure() 
plt.scatter(data, labels) 
plt.xlabel('X-axis') 
plt.ylabel('Y-axis') 
plt.title('Input data')

# 6 Define a deep nn with 2 hidden layers, where each hidden layer consists
# of 10 neurons and the output layer consists of one neuron
multilayer_net = nl.net.newff([[min_value, max_value]], [10, 10, 1]) 

# 7 Set the training algorithm to gradient descent:
multilayer_net.trainf = nl.train.train_gd 

# 8 Train the network
error = multilayer_net.train(data, labels, epochs=1000, show=100, lr=.001)

# 9 Predict the output for the training inputs to see the performance
predicted_output = multilayer_net.sim(data)

# 10 Plot the training error
plt.figure() 
plt.plot(error) 
plt.xlabel('Number of epochs') 
plt.ylabel('Error') 
plt.title('Training error progress')

# 11 Let's create a set of new inputs and run the neural network on them to see how it performs:
x2 = np.linspace(min_value, max_value, num_datapoints * 2) 
y2 = multilayer_net.sim(x2.reshape(x2.size,1)).reshape(x2.size) 
y3 = predicted_output.reshape(num_datapoints) 

# 12 Plot the output
plt.figure() 
plt.plot(x2, y2, '-', x, y, '.', x, y3, 'p') 
plt.title('Ground truth vs predicted output') 
plt.show() 
