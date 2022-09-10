# 1 Let's make the basic imports
import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

# 2 Define some input data to see where the datapoints are located
# Define input data
data = np.array([[.3, .2], [.1, .4], [.4, .6], [.9, .5]])
labels = np.array([[0], [0], [0], [1]])

# 3 Let's plot this data to see where the datapoints are located
# plot input data
plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('X-axis')
plt.xlabel('Y-axis')
plt.title('Input data')

# 4 Let's define a perceptron with 2 inputs. This function also needs us to specify
# the minimum and maximum values
# Define a perceptron with 2 inputs; 
# Each element of the list in the first argument  
# specifies the min and max values of the inputs 
perceptron = nl.net.newp([[0, 1],[0, 1]], 1)

# 5 Let's train the perceptron model:
# Train the perceptron 
error = perceptron.train(data, labels, epochs=50, show=15, lr=0.01) 

# 6 Let's plot the results
# plot results 
plt.figure() 
plt.plot(error) 
plt.xlabel('Number of epochs') 
plt.ylabel('Training error') 
plt.grid() 
plt.title('Training error progress') 
 
plt.show() 