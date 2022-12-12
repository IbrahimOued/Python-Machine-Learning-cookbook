# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# 2 Define a function to create a waveform, based on input parameters
def create_waveform(num_points):
    # Create train samples
    data1 = 1 * np.cos(np.arange(0, num_points))
    data2 = 2 * np.cos(np.arange(0, num_points))
    data3 = 3 * np.cos(np.arange(0, num_points))
    data4 = 4 * np.cos(np.arange(0, num_points))

    # 3 Create differents amplitutes for each interval to create a random waveform
    # Create varying amplitudes
    amp1 = np.ones(num_points)
    amp2 = 4 + np.zeros(num_points)
    amp3 = 2 + np.ones(num_points)
    amp4 = .5 + np.zeros(num_points)

    # 4 Combine the arrays to create output arrays. This data corresponds to the input and the
    # amplitude corresponds to the labels
    data = np.array([data1, data2, data3, data4]).reshape(num_points * 4, 1)
    amplitude = np.array([[amp1, amp2, amp3, amp4]]).reshape(num_points * 4, 1)
    return data, amplitude

# 5 Define a function to draw the output after passing the data through the trained nn
# Draw the output using the network
def draw_output(net, num_points_test):
    data_test, amplitude_test = create_waveform(num_points_test)
    output_test = net.sim(data_test)
    plt.plot(amplitude_test.reshape(num_points_test * 4))
    plt.plot(output_test.reshape(num_points_test * 4))

# 6 Define the main function and start by creating sample data
if __name__ == '__main__':
    # Get data
    num_points = 30
    data, amplitude = create_waveform(num_points)

    # 7 Create a rnn with 2 layers
    # Create network with 2 layers
    net = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

    # 8 Set the initialized functions for each layers
    # Set initialized functions and init
    net.layers[0].initf = nl.init.InitRand([-.1, .1], 'wb')
    net.layers[1].initf = nl.init.InitRand([-.1, .1], 'wb')
    net.init()

    # 9 Train the recurrent neural network
    # Training the recurrent neural network
    error = net.train(data, amplitude, epochs=1000, show=100, goal=.001)

    # 10 Compute the output from the neural network for the training data
    # compute the output
    output = net.sim(data)

    # 11 Plot the training error
    # Plot training results 
    plt.subplot(211) 
    plt.plot(error) 
    plt.xlabel('Number of epochs') 
    plt.ylabel('Error (MSE)')

    # 12 Plot the results
    plt.subplot(212) 
    plt.plot(amplitude.reshape(num_points * 4)) 
    plt.plot(output.reshape(num_points * 4)) 
    plt.legend(['Ground truth', 'Predicted output'])

    # 13 Create a waveform of random length and see whether the network can predict it:
    # TEsting on unknown data at multiple scales
    plt.figure()
    plt.subplot(211) 
    draw_output(net, 74) 
    plt.xlim([0, 300])

    # 14 Create another waveform with a shorter length and see whether the network can predict it:
    plt.subplot(212) 
    draw_output(net, 54) 
    plt.xlim([0, 300]) 
 
    plt.show()

    # you will see two diagrams. The first diagram displays training errors and the
    # performance on the training data, as follows:

    # The second diagram displays how a trained recurrent neural net performs on sequences of arbitrary lengths, as follows