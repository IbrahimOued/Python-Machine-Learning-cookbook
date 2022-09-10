# 1 Let's make the basic imports
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import wavfile  
from python_speech_features import mfcc, logfbank

# 2 Read the input_read.wav file
# Read input sound file
sampling_freq, audio = wavfile.read('./ch08/input_freq.wav')

# 3 Extract the MFCC and filter bank features, as follows:
mfcc_features = mfcc(audio, sampling_freq)
filterbank_features = logfbank(audio, sampling_freq)

# 4 Print the parameters to see how many windows were generated
# Print parameters 
print('MFCC:\nNumber of windows =', mfcc_features.shape[0])
print('Length of each feature =', mfcc_features.shape[1])
print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])

# 5 Let's now visualize the MFCC features. We need to transform the matrix so that time
# domain is horizontal
# Plot the features 
mfcc_features = mfcc_features.T 
plt.matshow(mfcc_features) 
plt.title('MFCC') 

# 6 Now, let's visualize the filter bank features. Again, we need to transform the 
# matrix so that the time domain is horizontal:
filterbank_features = filterbank_features.T 
plt.matshow(filterbank_features) 
plt.title('Filter bank') 
plt.show()

# 7 Let's run it 