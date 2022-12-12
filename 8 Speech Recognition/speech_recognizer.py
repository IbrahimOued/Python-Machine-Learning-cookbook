# 1 Let's make the basic imports

import os
import argparse

import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc

# 2 Define a function to parse the input arguments in the command line:
# Function to parse input arguments


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
                        help="Input folder containing the audio files in subfolders")
    return parser

# 3 Let's use the HMMTrainer class defined in the previous Building HMMs recipe:
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,
                                         covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

# Define the main function, and parse the input arguments:
if __name__=='__main__': 
    args = build_arg_parser().parse_args() 
    input_folder = args.input_folder

    # 5 Initiate the variable that will hold all the HMM models:
    hmm_models = []

    # 6 Parse the input directory that contains all the database's audio files:
    # Parse the input directory 
    for dirname in os.listdir(input_folder):
        # 7 Extract the name of the subfolder:
        # Get the name of the subfolder  
        subfolder = os.path.join(input_folder, dirname) 
 
        if not os.path.isdir(subfolder):  
            continue 

        # 8 Get the name of the subfolder  
        # Extract the label 
        label = subfolder[subfolder.rfind('/') + 1:]

        # 9 Initialize the variables for training
        # Initialize variables 
        X = np.array([]) 
        y_words = []

        # 10 Iterate through the list of audio files in each subfolder:
        # Iterate through the audio files (leaving 1 file for testing in each class) 
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            # 11 Read each audio file, as follows:
            # Read the input file 
            filepath = os.path.join(subfolder, filename) 
            sampling_freq, audio = wavfile.read(filepath) 

            # 12 Extract the MFCC features, as follows:
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)

            # 13 Keep appending this to the X variable, as follows
            # Append the MFCC features to the X variable
            if len(X) == 0: 
                X = mfcc_features 
            else: 
                X = np.append(X, mfcc_features, axis=0)

            # 14 Append the corresponding label too, as follows:
            # Append the label 
            y_words.append(label)
        
        # 15 Once you have extracted features from all the files in the current class,
        # train and save the HMM model. As HMM is a generative model for unsupervised 
        # learning, we don't need labels to build HMM models for each class.
        # We explicitly assume that separate HMM models will be built for each class: 
        # Train and save HMM model 
        hmm_trainer = HMMTrainer() 
        hmm_trainer.train(X) 
        hmm_models.append((hmm_trainer, label)) 
        hmm_trainer = None
    
    # 16 Get a list of test files that were not used for training:
    # Test files 
    input_files = [
            './ch08/data/pineapple/pineapple15.wav',
            './ch08/data/orange/orange15.wav', 
            './ch08/data/apple/apple15.wav', 
            './ch08/data/kiwi/kiwi15.wav' 
            ]

    # 17 Parse the input files, as follows:
    # Classify input data 
    for input_file in input_files:
        # 18 Read the input file, as follows:
        # Read the input file
        sampling_freq, audio = wavfile.read(input_file)

        # 19 Extract the MFCC features, as follows:
        # Extract MFCC features
        mfcc_features = mfcc(audio, sampling_freq)

        # 20 Define the variables to store the maximum score and the output label:
        # Define variables 
        max_score = float('-inf')
        output_label = None

        # 21 Iterate through all the models and run the input file through each of them:
        # Iterate through all HMM models and pick  
        # the one with the highest score 
        for item in hmm_models: 
            hmm_model, label = item

            # 22 Extract the score and store the maximum score:
            score = hmm_model.get_score(mfcc_features) 
            if score > max_score: 
                max_score = score 
                output_label = label
            
        # 23 Print the true and predicted labels:
        # Print the output
        print("True:", input_file[input_file.find('/')+1:input_file.rfind('/')])
        print("Predicted:", output_label)

    # 24 Run the code like this $ python ./ch08/speech_recognizer.py --input-folder ./ch08/data 

        
