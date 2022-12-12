# 1 Let's import the basic files
from sklearn import model_selection
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 2 Load the dataset
input_file = "ch02/car.data.txt"
# reading the data
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

# Each line contains a comma-separated list of words.
# Therefore, we parse the input file, split each line,
# and then append the list to the main data. We ignore
# the last character on each line because it's a newline
# character. Python packages only work with numerical
# data, so we need to transform these attributes
# into something that those packages will understand.

# 3 In the previous chapter, we discussed label encoding. That
# is what we will use here to convert strings to numbers:

# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# As each attribute can take a limited number of values,
# we can use the label encoder to transform them into
# numbers. We need to use different label encoders for
# each attribute. For example, the lug_boot attribute
# can take three distinct values, and we need a
# label encoder that knows how to encode this attribute.
# The last value on each line is the class, so we assign it to the y variable.

# 4 Let's train the classifier
# Build a Random Forest classifier
params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)
# ou can play around with the n_estimators and max_depth
# parameters to see how they affect classification accuracy.
# We will actually do this soon in a standardized way.

# 5 Let's perform cross validation
# Cross validation
accuracy = model_selection.cross_val_score(
    classifier, X, y, scoring='accuracy', cv=3)
print("Accuracy of the classifier: " + str(round(100*accuracy.mean(), 2)) + '%')

# Once we train the classifier, we need to see how it performs. We use
# three-fold cross-validation to calculate the accuracy here.

# 6 One of the main goals of building a classifier is to use it on isolated unknown
# data instances. Let's use a single datapoint and see how we can use this classifier to
# categorize it
# Testing encoding on single data instance
input_data = ['high', 'low', '2', 'more', 'med', 'high']
input_data_encoded = [-1] * len(input_data)
for i, item in enumerate(input_data):
    input_data_encoded[i] = int(label_encoder[i].transform([input_data[i]]))
input_data_encoded = np.array(input_data_encoded)

# The first step was to convert that data into numerical data. We need to use the label
# encoders that we used during training because we want it to be consistent. If there
# are unknown values in the input datapoint, the label encoder will complain because it
# doesn't know how to handle that data. For example, if you change the first value in
# the list from high to abcd, then the label encoder won't work because
# it doesn't know how to interpret this string. This acts like an error check to see
# whether the input datapoint is valid.

# 7 We are now ready to predict the output class of this datapoint
# Predict and print output for a particular datapoint
output_class = classifier.predict([input_data_encoded])
print("Output class:", label_encoder[-1].inverse_transform(output_class)[0])
# We use the predict() method to estimate the output class. If we output the
# encoded output label, it won't mean anything to us. Therefore, we use the
# inverse_transform method to convert this label back to its original form
# and print out the output class.

# Validation curves
classifier = RandomForestClassifier(max_depth=4, random_state=7)

parameter_grid = np.linspace(25, 200, 8).astype(int)
train_scores, validation_scores = validation_curve(
    estimator=classifier, X=X, y=y, param_name='n_estimators', param_range=parameter_grid, cv=5)
print("##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", train_scores)
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)

# In this case, we defined the classifier by fixing the max_depth
# parameter. We want to estimate the optimal number of estimators to use,
# and so have defined our search space using parameter_grid. It is going to extract training and
# validation scores by iterating from 25 to 200 in 8 steps.

# 3 Run it

# 4 Let's plot it
# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Training curve')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

# 5 Let's do the same for the max_depth parameter
classifier = RandomForestClassifier(
    n_estimators=20, random_state=7)
parameter_grid = np.linspace(2, 10, 5).astype(int)

train_scores, valid_scores = validation_curve(
    estimator=classifier, X=X, y=y, param_name='n_estimators', param_range=parameter_grid, cv=5)
print("\nParam: max_depth\nTraining scores:\n", train_scores)
print("\nParam: max_depth\nValidation scores:\n", validation_scores)

# We fixed the n_estimators parameter at 20 to see how the performance varies
# with max_depth. Here is the output on the Terminal:
# 6 Let's plot it
# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Validation curve')
plt.xlabel('Maximum depth of the tree')
plt.ylabel('Accuracy')
plt.show()


# 7 Let's run


## Learning curves
# 1
classifier = RandomForestClassifier(random_state=7)
parameter_grid = np.array([200, 500, 800, 1100])
train_scores, validation_scores = validation_curve(
    estimator=classifier, X=X, y=y, param_name='n_estimators', param_range=parameter_grid, cv=5)
print("\n##### LEARNING CURVES #####")
print("\nTraining scores:\n", train_scores)
print("\nValidation scores:\n", validation_scores)

# We want to evaluate the performance metrics using training datasets of 200, 500, 800, and 1,100 
# samples. We use five-fold cross-validation, as specified by the cv parameter in the validation_curve method.

# 2 If we run the code

# 3 Let's plot
# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()

# 4 See the output
# Although smaller training sets seem to give better accuracy, they are prone to overfitting.
# If we choose a bigger training dataset, it consumes more resources. Therefore, we need to make
# a trade-off here to pick the right size for the training dataset.