# 1 Import the basic packages
from email.mime import audio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 2 We will use the wavfile package to read the audio from the input_read.wav file
# read the input file
sampling_freq, audio = wavfile.read('./ch08/input_read.wav')

# 3 Let's print out the parameters of this signal
# print the params
print("Shape: ", audio.shape)
print("Datatype: ", audio.dtype)
print("Duration: ", round(audio.shape[0] / float(sampling_freq), 3), 'seconds')

# 4 The audio signal is stored as 16 bit signed integer data, we need to NORMALIZE these values
# Normalize the values
audio = audio / (2.**15)

# 5 Now, let's extract the first 30 Values to plot
# Extract first 30 values for plotting
audio = audio[:30]

# 6 The x axis is the time axis. Let's build this axis, considering the fact that it
# should be scaled using the sampling frequency factor
# Build the time axis
x_values = np.arange(0, len(audio), 1) / float(sampling_freq)

# 7 Convert the units to seconds
# convert to seconds
x_values *= 1000

# 8 Let's now plot this features
# Plotting the chopped audio signal 
plt.plot(x_values, audio, color='black') 
plt.xlabel('Time (ms)') 
plt.ylabel('Amplitude') 
plt.title('Audio signal') 
plt.show()

# 9 Let's run it