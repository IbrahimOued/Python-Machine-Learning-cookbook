# 1 Let's make the basic imports
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io.wavfile import write 

# 2 We need to define the output file where the generated audio will be stored:
# File where the output will be saved 
output_file = 'output_generated.wav' 

# 3 Let's now specify the audio generation parameters. We want to generate a 
# 3-second long signal with a sampling frequency of 44,100, and a tonal frequency of 587 Hz.
# The values on the time axis will go from -2*pi to 2*pi:
# Specify audio parameters 
duration = 3  # seconds 
sampling_freq = 44100  # Hz 
tone_freq = 587 
min_val = -2 * np.pi 
max_val = 2 * np.pi

# 4 Let's generate the time axis and the audio signal. The audio signal is a simple
# sinusoid with the previously mentioned parameters:
# Generate audio 
t = np.linspace(min_val, max_val, duration * sampling_freq) 
audio = np.sin(2 * np.pi * tone_freq * t)

# 5 Now, let's add some noise to the signal:
# Add some noise 
noise = 0.4 * np.random.rand(duration * sampling_freq) 
audio += noise

# 6 We need to scale the values to 16-bit integers before we store them:
# Scale it to 16-bit integer values 
scaling_factor = pow(2,15) - 1 
audio_normalized = audio / np.max(np.abs(audio)) 
audio_scaled = np.int16(audio_normalized * scaling_factor)

# 7 Write this signal to the output file:
# Write to output file 
write(output_file, sampling_freq, audio_scaled) 

# 8 Plot the signal using the first 100 values:
# Extract first 100 values for plotting 
audio = audio[:100]

# 9 Generate the time axis, as follows:
# Build the time axis 
x_values = np.arange(0, len(audio), 1) / float(sampling_freq) 

# 10 Convert the time axis into seconds:
# Convert to seconds 
x_values *= 1000 

# 11 Plot the signal, as follows:
# Plotting the chopped audio signal 
plt.plot(x_values, audio, color='black') 
plt.xlabel('Time (ms)') 
plt.ylabel('Amplitude') 
plt.title('Audio signal') 
plt.show()

# 12 Let's run it