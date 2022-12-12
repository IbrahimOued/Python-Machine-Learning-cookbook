# 1 Make the basic imports
import json 
import numpy as np 
from scipy.io.wavfile import write 

# 2 Define a function to synthesize a tone, based on input params
# Synthesize tone 
def synthesizer(freq, duration, amp=1.0, sampling_freq=44100):

    # 3 Biild the time axis values
    # Build the time axis 
    t = np.linspace(0, duration, round(duration * sampling_freq)) 

    # 4 Construct the audio sample using the input arguments, such as amplitude and frequency
    # Construct the audio signal 
    audio = amp * np.sin(2 * np.pi * freq * t) 
 
    return audio.astype(np.int16)

# 5 Let's define the main function. We've been provided a json file, tone_freq_map.json
# which contains some notes along with their frequencies
if __name__=='__main__': 
    tone_map_file = './ch08/tone_freq_map.json' 

    # 6 Load that file
    # Read the frequency map 
    with open(tone_map_file, 'r') as f: 
        tone_freq_map = json.loads(f.read())

    # 7 Now, let's assume that we want to generate a G note for a duration of two seconds
    # Set input parameters to generate 'G' tone 
    input_tone = 'G' 
    duration = 2     # seconds 
    amplitude = 10000 
    sampling_freq = 44100    # Hz

    # 8 Call the function with the following parameters:
    # Generate the tone 
    synthesized_tone = synthesizer(tone_freq_map[input_tone], duration, amplitude, sampling_freq) 
    # 9 Write the generated signal into the output file, as follows:
    # Write to the output file 
    write('./ch08/output_tone.wav', sampling_freq, synthesized_tone)

    # 10 Now, let's do something more interesting. Let's generate some notes in 
    # sequence to give it a musical feel. Define a note sequence along with their durations in seconds:
    # Tone-duration sequence 
    tone_seq = [('D', 0.3), ('G', 0.6), ('C', 0.5), ('A', 0.3), ('Asharp', 0.7)]

    # 11 Iterate through this list and call the synthesizer function for each of them:
    # Construct the audio signal based on the chord sequence 
    output = np.array([]) 
    for item in tone_seq: 
        input_tone = item[0] 
        duration = item[1] 
        synthesized_tone = synthesizer(tone_freq_map[input_tone], duration, amplitude, sampling_freq) 
        output = np.append(output, synthesized_tone, axis=0) 
    output = output.astype(np.int16)

    # 12 Write the signal to the output file:
    # Write to the output file 
    write('./ch08/output_tone_seq.wav', sampling_freq, output)

    # 13 You can now open the output_tone_seq.wav file in your media player and listen to it. You can feel the music!