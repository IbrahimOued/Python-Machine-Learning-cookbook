# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import utilities

# 2 Let's load the data (data_multivar_inbalanced.txt)
input_file = 'ch03/data_multivar_imbalance.txt'
X, y = utilities.load_data(input_file)

# 3 Let's visualize the data.
# Separate the data into classes based on 'y'
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])
# Plot the input data
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None', edgecolors='black', marker='s')
plt.title('Input data')
plt.show()

# 4 And let's run it

# 5 Let's build an SVM with a linear kernel. The code is the same as it was in the previous recipe,
# Building a nonlinear classifier using SVMs
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.25, random_state=5)
# params = {'kernel':'linear'} For the balanced case
params = {'kernel': 'linear', 'class_weight': 'balanced'}  # For the imbalanced case

classifier = SVC(**params, gamma='auto')
classifier.fit(X_train, y_train)
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()

# 6 Let's print a classification report
from sklearn.metrics import classification_report
target_names = ['Class-' + str(int(i)) for i in set(y)]
print("\n" + "#"*30)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
print("#"*30 + "\n")
print("#"*30)
print("\nClassification report on test dataset\n")
print(classification_report(y_test, classifier.predict(X_test), target_names=target_names))
print("#"*30 + "\n")

# 7 After the run we can see the result

# 8 You might wonder why there's no boundary here! Well, 
# this is because the classifier is unable to separate the 
# two classes at all, resulting in 0% accuracy for Class-0. 
# You will also see a classification report printed on your Terminal
# 9 As we expected, Class-0 has 0% precision, so let's go ahead and 
# fix this! In the Python file, search for the following line:
# params = {'kernel': 'linear'}
# 10 And let's replace it with
# params = {'kernel': 'linear', 'class_weight': 'balanced'}
# 11 The class_weight parameter will count the number of datapoints in each 
# class to adjust the weights so that the imbalance doesn't adversely affect the performance.
# 12 Run to see the output
# 13 And also the classification report
# 14 As we can see, Class-0 is now detected with nonzero percentage accuracy.