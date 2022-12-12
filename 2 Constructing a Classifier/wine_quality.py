# 1 Let's import the usual librairies and the dataset
import numpy as np
input_file='ch02/wine.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[1:])
        y.append(data[0])
    X = np.array(X)
    y = np.array(y)
# Two arrays are returned x(input data) and y(target)

# 2 Now we need to separate our data into two groups:
# a training dataset and a testing dataset. The training dataset will
# be used to build the model, and the testing dataset
# will be used to see how this trained model performs on unknown data:
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
# Four arrays are returned: X_train, X_test, y_train, and y_test.
# This data will be used to train and validate the model.

# 3 Let's train the classifier
from sklearn.tree import DecisionTreeClassifier
classifier_DecisionTree = DecisionTreeClassifier()
classifier_DecisionTree.fit(X_train, y_train)

# To train the model, a decision tree algorithm has been used. A decision tree algorithm is
# based on a non-parametric supervised learning method used for classification and regression.
# The aim is to build a model that predicts the value of a target variable using decision
# rules inferred from the data features.

# 4 Now it's time to the compute accuracy of the classifier:
y_test_pred = classifier_DecisionTree.predict(X_test)
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")

# 5 Finally, a confusion matrix will be calculated to compute the model performance:
from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test, y_test_pred)
print(confusion_mat)
# Values not present on the diagonals represent classification errors. So, only four errors were committed by the classifier.