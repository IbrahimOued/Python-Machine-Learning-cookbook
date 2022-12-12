# 1 Create a file called regressor.py and add the following ligne
filename = "ch01/VehiclesItaly.txt"
X = []
y = []
with  open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

# We just loaded the input data into X and y, where X refers to the
# independent variable (explanatory variables) and y refers to the
# dependent variable (response variable). Inside the loop in the
# preceding code, we parse each line and split it based on the comma
# operator. We then convert them into floating point values and save them in X and y

# 2 When we build a ml model, we need a way to validate our model and check
# wheter it is performing at a satisfactory level. To do this, we need to separate our data
# into 2 groups (training dataset and testing dataset). The testing dataset will be used to build
# the model, and the testing dataset will be used to see how this trained model performs on unknown data.
# So, let's go and split this data

num_training = int(.8 * len(X))
num_test = len(X) - num_training

import numpy as np
# Training data
X_train = np.array(X[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])

# Test data
X_test = np.array(X[:num_test]).reshape((num_test, 1))
y_test = np.array(y[:num_test])

# First, we have put aside 80% of the data for the training dataset
# and the remaining 20% is for the testing dataset. Then, we have
# built four arrays: X_train, X_test,y_train, and y_test.

# 3 We are now ready to train the model. Let's create a regressor object, as follows:
from sklearn import linear_model

# Create linear regression object
linear_regressor=linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

# 4 We just trained the linear regressor, based on our training data.
# The fit() method takes the input data and trains the model.
# To see how it all fits, we have to predict the training data with the model fitted:

y_train_pred = linear_regressor.predict(X_train)

# 5 To plot the outputs, we will use the matplotlib library as follows:
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()

# 6 In the preceding code, we used the trained model to predict the output for our training data.
# This wouldn't tell us how the model performs on unknown data, because we are
# running it on the training data. This just gives us an idea of how the model fits on training data.
# Looks like it's doing okay, as you can see in the preceding diagram!

# 7 Let's predict the test dataset output based on this model and plot it, as follows:
y_test_pred = linear_regressor.predict(X_test)
plt.figure()
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()

#============= Computing regression accuracy=============
# 1 Now we will use the functions available to evaluate the performance of the linear
# regression model we developed in the previous recipe
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# n R2 score near 1 means that the model is able to predict the data very well.
# Keeping track of every single metric can get tedious, so we pick one or
# two metrics to evaluate our model. A good practice is to make sure that the mean
# squared error is low and the explained variance score is high.

# Achieving model persistance
# 1 Add the following lines to the regressor.py file
import pickle
output_model_file = "ch01/3_model_linear_regr.pkl"
with open(output_model_file, 'wb') as f:
    pickle.dump(linear_regressor, f)

# 2 The regressor object will be saved in the saved_model.pkl file. Let's look at how to load it and use it, as follows:

with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print("New mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
# Here, we just loaded the regressor from the file into the model_linregr variable.
# You can compare the preceding result with the earlier result to confirm that it's the same

# Ridge regressor
# 1 You can use the data already used in the previous example
# Add the following lines
# from sklearn import linear_model
ridge_regressor = linear_model.Ridge(alpha=.01, fit_intercept=True, max_iter=10000)
# 3 The alpha parameter controls the complexity. As alpha gets closer to 0, the ridge regressor tends
# to become more like a linear regressor with ordinary least squares. So, if you want to make it robust
# against outliers, you need to assign a higher value to alpha. We considered a value of .01, which is moderate
# 4 Let's train this regressor, as follows:
ridge_regressor.fit(X_train, y_train)
y_test_pred_ridge = ridge_regressor.predict(X_test)
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge), 2))
