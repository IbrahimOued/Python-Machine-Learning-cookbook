# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt

import utilities

# 2 We just imported a couple of packages and named
# the input file. Let's look at the load_data() method
# Load the input data
input_file = 'ch03/data_multivar.txt'
X, y = utilities.load_data(input_file)

# 3 We need to separate the data into classes, as follows:
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0 ])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1 ])

# 4 Now that we have separated the data, let's plot it
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], facecolors='none', edgecolors='black', marker='s')
plt.title('Input data')
plt.show()

"""
The preceding consists of two types of points—solid squares and 
empty squares. In machine learning lingo, we say that our data
consists of two classes. Our goal is to build a model that can separate the solid squares from the empty squares.
"""

# (Suite)
# 1 We need to split our dataset into training and testing datasets
# Train test split and SVM training
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=42)

# 2 Let's initialize the SVM object using linear kernel.
# params = {'kernel' : 'linear'}
# params =  {'kernel':'poly'} # For the polynomial kernel
params =  {'kernel':'rbf'} # For the radial basis function kernel
classifier = SVC(**params, gamma='auto')

# 3 We are now ready to train the linear SVM classifier
classifier.fit(X_train, y_train)

# 4 We can now se how the classifier performs
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset') 
plt.show() 

# 5 Let's see how this performs on the test dataset
y_test_pred = classifier.predict(X_test) 
utilities.plot_classifier(classifier, X_test, y_test, 'Test dataset') 
plt.show()

# As we can see, the classifier boundaries on the input data are clearly identified.

# 6 Let's compute the accuracy for the training set.
from sklearn.metrics import classification_report

target_names = ['Class-' + str(int(i)) for i in set(y)]
print("\n" + '#'*30)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
print("#"*30 + "\n")
# We can run it

# 7 Finally, let's see the classification report for the testing dataset:
print("#"*30)
print("\nClassification report on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print("#"*30 + "\n")

# 8 If you run this code, you will see the following on the Terminal:
"""
From the output screenshot where we visualized the data, we can see that the solid
squares are completely surrounded by empty squares.
This means that the data is not linearly separable. We cannot draw a nice straight line
to separate the two sets of points! Hence, we need a nonlinear classifier to separate these datapoints.
"""

# Building a nonlinear classifier using SVMs

# 1 The first case, let's use a polynomial kernel to build a non linear classifier

# Replace params =  {'kernel':'linear'} by params =  {'kernel':'poly'}
