# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# 2 We will use the data in the data_single_layer.txt
# Define input data
input_file = 'ch13/data_single_layer.txt'
input_text = np.loadtxt(input_file)
data = input_text[:, 0:2]
labels = input_text[:, 2:]

# 3 Let's plot the input data
plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('X-axis')
plt.xlabel('Y-axis')
plt.title('Input data')

# 4 Let's extract the minimum and maximum value
# Min and max values for each dimension
x_min, x_max = data[:, 0].min(), data[:, 0].max()
y_min, y_max = data[:, 1].min(), data[:, 1].max()

# 5 Let's define a single layer neural network with 2 neurons in the hidden layer
# Define a single layer neural network with 2 neurons
# Each element in the list(first argument) specifies the
# min and max values of the inputs
single_layer_net = nl.net.newp([[x_min, x_max], [y_min, y_max]], 2)

# 6 Train the neural network for 50 epochs
# Train the neural network
error = single_layer_net.train(data, labels, epochs=50, show=20, lr=.001)

# 7 Plot the results
# Plot results 
plt.figure() 
plt.plot(error) 
plt.xlabel('Number of epochs') 
plt.ylabel('Training error') 
plt.title('Training error progress') 
plt.grid() 
plt.show() 

# 8 Let's test the neural network on the new data
print(single_layer_net.sim([[0.3, 4.5]]))
print(single_layer_net.sim([[4.5, 0.5]]))
print(single_layer_net.sim([[4.3, 8]]))