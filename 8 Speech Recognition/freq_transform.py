# 1 Make the basic imports
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# 2 Read the input_freq.wav file
# Read the input file
sampling_freq, audio = wavfile.read('./ch08/input_freq.wav')

# 3 Normalize the signal
# Normalize the values
audio = audio / (2.**15)

# 4 The audio signal is just a Numpy array. So, you can extract the
# length using the following code
# extract length
len_audio = len(audio)

# 5 Let's apply the Fourier transform. The Fourier transform signal is
# mirrored along the center, so we just need to take the first half of
# the transformed signal. Our end goal is to extract the power 
# signal, so we square the values in the signal in preparation for this:
# Apply Fourier transform 
transformed_signal = np.fft.fft(audio) 
half_length = np.ceil((len_audio + 1) / 2.0) 
transformed_signal = abs(transformed_signal[0:int(half_length)]) 
transformed_signal /= float(len_audio) 
transformed_signal **= 2

# 6 Extract the length of the signal, as follows:
# Extract length of transformed signal 
len_ts = len(transformed_signal)

# 7 We need to double the signal according to the length of the signal:
# Take care of even/odd cases 
if len_audio % 2: 
    transformed_signal[1:len_ts] *= 2 
else: 
    transformed_signal[1:len_ts-1] *= 2 

# 8 The power signal is extracted using the following formula:
# Extract power in dB 
power = 10 * np.log10(transformed_signal) 

# 9 The x axis is the time axis; we need to scale this according to the
# sampling frequency and then convert this into seconds:
# Build the time axis 
x_values = np.arange(0, half_length, 1) * (sampling_freq / len_audio) / 1000.0

# 10 Plot the signal, as follows:
# Plot the figure 
plt.figure() 
plt.plot(x_values, power, color='black') 
plt.xlabel('Freq (in kHz)') 
plt.ylabel('Power (in dB)') 
plt.show()

# 11 And run it
