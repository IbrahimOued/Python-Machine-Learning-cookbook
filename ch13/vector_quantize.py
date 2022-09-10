# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# 2 Let's load the input data from the data_vq.txt file
input_file = 'ch13/data_vq.txt'
input_text = np.loadtxt(input_file)
data = input_text[:, 0:2]
labels = input_text[:, 2:]

# 3 Define a learning vector quantization (LVQ) neural network with 2 layers
# the array in the last parameter specifies the percentatge weightage for each ouput
# (they should add up to 1)
net = nl.net.newlvq(nl.tool.minmax(data), 10, [.25, .25, .25, .25])

# 4 Train the LVQ neural network
error = net.train(data, labels, epochs=100, goal=.1)

# 5 Create a grid of values for testing and visualization
xx, yy = np.meshgrid(np.arange(0, 8, .2), np.arange(0, 8, .2))
xx.shape = xx.size, 1
yy.shape = yy.size, 1
input_grid = np.concatenate((xx, yy), axis=1)

# 6 Evaluate the network on this grid
output_grid = net.sim(input_grid)

# 7 Define the four classes in our data
class1 = data[labels[:, 0] == 1]
class2 = data[labels[:, 1] == 1]
class3 = data[labels[:, 2] == 1]
class4 = data[labels[:, 3] == 1]

# 8 Define the grids for all these classes
grid1 = input_grid[output_grid[:, 0] == 1]
grid2 = input_grid[output_grid[:, 1] == 1]
grid3 = input_grid[output_grid[:, 2] == 1]
grid4 = input_grid[output_grid[:, 3] == 1]

# 9 Plot the outputs
plt.plot(class1[:, 0], class1[:, 1], 'ko', class2[:, 0], class2[:, 1], 'ko',
         class3[:, 0], class3[:, 1], 'ko', class4[:, 0], class4[:, 1], 'ko')
plt.plot(grid1[:, 0], grid1[:, 1], 'b.', grid2[:, 0], grid2[:, 1], 'gx',
         grid3[:, 0], grid3[:, 1], 'cs', grid4[:, 0], grid4[:, 1], 'ro')
plt.axis([0, 8, 0, 8])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Vector quantization using neural networks')
plt.show()
# you will see the following diagram, where the space is divided into regions.
# Each region corresponds to a bucket in the list of vector-quantized regions in the space:



