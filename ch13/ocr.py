# 1 Let's make the basic imports
import numpy as np
import neurolab as nl

# 2 Define the input filename
input_file = 'ch13/letter.data'

# 3 When we work with neural networks that deal with large amounts of data
# it takes a lot of time to train. To demonstrate how to build this system
# we take only 20 datapoints
num_datapoints = 20

# 4 If you look at the data, you will see that there are seven distinct characters in the first 20 lines
# Let's define them
orig_labels = 'omandig'
num_output = len(orig_labels)

# 5 We will use 90% of the data for training and the reamining 10% for testing. Define the training and testing
# parameters
num_train = int(.9 * num_datapoints)
num_test = num_datapoints - num_train

# 5 The starting and ending indices in each line of the dataset file are specified
start_index = 6
end_index = -1

# 6 Create the dataset
data = [] 
labels = [] 
with open(input_file, 'r') as f: 
    for line in f.readlines(): 
        # Split the line tabwise 
        list_vals = line.split('\t')

        # 8 Add an error check to see whether the 
        # characters are in our list of labels
        # (if the label is not in our ground truth labels, skip it):
        if list_vals[1] not in orig_labels: 
            continue
        # 9 Extract the label, and append it to the main list
        label = np.zeros((num_output, 1)) 
        label[orig_labels.index(list_vals[1])] = 1 
        labels.append(label)

        # 10 Extract the character, and append it to the main list:
        cur_char = np.array([float(x) for x in list_vals[start_index:end_index]])
        data.append(cur_char)

        # 11 Exit the loop once we have enough data
        if len(data) >= num_datapoints: 
            break
        # 12 Convert this data into NumPy arrays:
        data = np.asfarray(data) 
        labels = np.array(labels).reshape(num_datapoints, num_output)

        # 13 Extract the number of dimensions in our data
        num_dims = len(data[0])

        # 14 Train the nn until 10.000 epochs
        net = nl.net.newff([[0, 1] for _ in range(len(data[0]))], [128, 16, num_output])
        net.trainf = nl.train.train_gd
        error = net.train(data[:num_train, :], labels[:num_train, :], epochs=10000, show=100, goal=.01)

        # 15 Predict the output for test inputs
        predicted_output = net.sim(data[num_train:, :])
        print("Testing on unknown data:")
        for i in range(num_test):
            print("Original:", orig_labels[np.argmax(labels[i])])
            print("Predicted:", orig_labels[np.argmax(predicted_output[i])])