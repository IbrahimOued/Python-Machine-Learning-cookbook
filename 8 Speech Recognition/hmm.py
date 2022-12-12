# 1 Let's make the basic imports and create a
import numpy as np
# Class to handle all HMM related processing 
class HMMTrainer(object): 
    # 2 Let's initialize the class; we will use Gaussian HMMs to model our data.
    # The n_components parameter defines the number of hidden states. cov_type
    # defines the type of covariance in our transition matrix, and n_iter indicates
    # the number of iterations it will go through before it stops training:
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000): 
        # The choice of the preceding parameters depends on the problem at hand.
        # You need to have an understanding 
        # of your data in order to select these parameters in a smart way.

        # 3 Initialize the variables, as follows:
        self.model_name = model_name 
        self.n_components = n_components 
        self.cov_type = cov_type 
        self.n_iter = n_iter 
        self.models = [] 

        # 4 Define the model with the following parameters:
        if self.model_name == 'GaussianHMM': 
            self.model = hmm.GaussianHMM(n_components=self.n_components,  
                    covariance_type=self.cov_type, n_iter=self.n_iter) 
        else: 
            raise TypeError('Invalid model type')

    # 5 The input data is a NumPy array, where each element is a feature
    # vector consisting of k dimensions:
    # X is a 2D numpy array where each row is 13D 
    def train(self, X): 
        np.seterr(all='ignore') 
        self.models.append(self.model.fit(X))

    # 6 Define a method to extract the score, based on the model:
    # Run the model on input data 
    def get_score(self, input_data): 
        return self.model.score(input_data)

    # 7 We built a class to handle HMM training and prediction, but we need
    # some data to see it in action.
    # We will use it in the next recipe to build a speech recognizer. 