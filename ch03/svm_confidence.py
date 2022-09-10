# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import utilities

# Load input data
input_file = 'ch03/data_multivar.txt'
X, y = utilities.load_data(input_file)

# 2 At this point, we split 
# the data for training and testing, and then we will build the classifier:
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
params = {'kernel': 'rbf'}
classifier = SVC(**params, gamma='auto')
classifier.fit(X_train, y_train)

# 3 Define the input datapoint:
input_datapoints = np.array([[2, 1.5], [8, 9], [4.8, 5.2], [4, 4], [2.5, 7], [7.6, 2], [5.4, 5.9]])

# 4 Let's measure the distance from the boundary:
print("Distance from the boundary:")
for i in input_datapoints:
    print(i, '-->', classifier.decision_function([i])[0])

# 5 Check the terminal and see what's printed
# It's the distance from the boundary

# 6 The distance from the boundary gives us some information 
# about the datapoint, but it doesn't exactly tell us how confident 
# the classifier is about the output tag. To do this, we need 
# Platt scaling. This is a method that converts the distance measure
# into a probability measure between classes. Let's go ahead and 
# train an SVM using Platt scaling:
# Confidence measure 
params = {'kernel': 'rbf', 'probability': True} 
classifier = SVC(**params, gamma='auto')
# The probability parameter tells the SVM that it should train to compute the probabilities as well.

# 7 Let's train the classifier
classifier.fit(X_train, y_train)

# 8 Let's compute the confidence measurements for these input datapoints:print("Confidence measure:")
for i in input_datapoints:
    print(i, '-->', classifier.predict_proba([i])[0])
# The predict_proba function measures the confidence value.

# 9 We'll see in the terminal the probabilities of each point to be on either
# 10 Let's see where the points are with respect to the boundary:
utilities.plot_classifier(classifier, input_datapoints, [0]*len(input_datapoints), 'Input datapoints', 'True') 

# 11 Let's run it


