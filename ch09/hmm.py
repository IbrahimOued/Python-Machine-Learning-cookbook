# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# 2 We will use the data from a file named data_hmm.txt that
# is already provided to you. This file contains comma-separated lines.
# Each line contains three values: a year, a month, and a piece of
# floating-point data. Let's load this into a NumPy array
# Load data from input file 
input_file = 'ch09/data_hmm.txt' 
data = np.loadtxt(input_file, delimiter=',')

# 3 Let's tack the data column-wise for analysis. We don't
# need to technically column-stack this
# because it's only one column. However, if you have more
# than one column to analyze, you can use the following structure
# Arrange data for training
X = np.column_stack([data[:,2]])

# 4 Create and train the HMM using 4 components.
# The number of components is a hyperparameter that we
# have to choose. Here, by selecting four, we say that the data
# is being generated using four underlying states.
# We will see how the performance varies with this parameter:
# Create and train Gaussian HMM  
print("Training HMM....") 
num_components = 4 
model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000) 
model.fit(X) 

# 5 Run the predictor to get the hidden states:
# Predict the hidden states of HMM  
hidden_states = model.predict(X) 

# 6 Compute the mean and varianve of the hidden states
print("Means and variances of hidden states:")
for i in range(model.n_components):
    print("Hidden state", i+1)
    print("Mean =", round(model.means_[i][0], 3))
    print("Variance =", round(np.diag(model.covars_[i])[0], 3))

# As we discussed earlier, HMMs are generative models. So, let's generate, for example, 1000 samples and plot this
# Generate data using model 
num_samples = 1000 
samples, _ = model.sample(num_samples)  
plt.plot(np.arange(num_samples), samples[:,0], c='black') 
plt.title('Number of components = ' + str(num_components)) 
plt.show() 

# You can experiment with the n_components parameter to see how the curve gets
# nicer as you increase it. You can basically give it more freedom to train and
# customize by allowing a larger number of hidden states. If you increase it to 8,

