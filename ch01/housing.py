# 1 Import the librairies
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# 2 There is a standard housing dataset that people tend to use to get started with
# machine learning. You can download it at https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
housing_data = datasets.load_boston()

# 3 Let's sepate this into input and output. To make this independant of the ordering
# of the data, let's shuffle it as well
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# The sklearn.utils.shuffle() function shuffles arrays or sparse matrices in a consistent way
# to do random permutations of collections. Shuffling data reduces variance and makes sure
# that the patterns remain general and less overfitted. The random_state parameter controls
# how we shuffle data so that we can have reproducible results. 

# 4 Let's divide the data into training and testing. We'll allocate 80% for training and 20% for testing
num_training = int(.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[:num_training:], y[:num_training:]

# Remember, machine learning algorithms, train models by using a finite set of training data. In the training phase,
# the model is evaluated based on its predictions of the training set. But the goal of the algorithm is to produce a
# model that predicts previously unseen observations, in other words, one that is able to generalize the problem by
# starting from known data and unknown data.
# For this reason, the data is divided into two datasets: training and test. The training set is used to train the model,
# while the test set is used to verify the ability of the system to generalize.

# 5 We are now ready to fit a decision tree regression model. Let's pick a tree with a
# maximum depth of 4, which means that we are not letting the tree become aribtrary deep

dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train, y_train)

# The DecisionTreeRegressor function has been used to build a decision tree regressor

# 6 Let's also fit the decision tree regression model with Adaboost

ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ab_regressor.fit(X_train, y_train)

# The Adaboost function has been used to compare the results and see how Adaboost really
# boosts the performance of a decision tree regressor

# 7 Let's evaluate the perf of the decision tree regressor
y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)
print("#### Decision Tree performance ####")
print('Mean square error = ', round(mse, 2))
print('Explained variance score=', round(evs, 2))

# First, we used the predict() function to predict the response 
# variable based on the test data. Next, we calculated mean squared error
# and explained variance. Mean squared error is the average of the squared difference
# between actual and predicted values across all data points in the input.
# The explained variance is an indicator that, in the form of proportion, 
# indicates how much variability of our data is explained by the model 
# in question.

# 8 Now, let's evaluate the performance of AdaBoost
y_pred_ab = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)
print("#### AdaBoost performance ####")
print('Mean square error = ', round(mse, 2))
print('Explained variance score=', round(evs, 2))

# The error is lower and the variance score is closer to 1 when we use AdaBoost,
# as shown in the preceding output.

# Compute the relative importance of features
# 1 Let's see how to extract this.
DTFImp = dt_regressor.feature_importances_
DTFImp = 100.0 * (DTFImp / max(DTFImp))
index_sorted = np.flipud(np.argsort(DTFImp))
pos = np.arange(index_sorted.shape[0] + .5)

# he regressor object has a callable feature_importances_ method that gives
# us the relative importance of each feature. To compare the results, the
# importance values have been normalized. Then, we ordered the index values
# and turned them upside down so that they are arranged in descending order of importance. Finally, for 
# display purposes, the location of the labels on the x-axis has been centered.

# 2 To visualize the results, let's plot a bar graph
plt.figure()
plt.bar(pos, DTFImp[index_sorted], align='center')
plt.xticks(pos, housing_data.feature_names[index_sorted])
plt.ylabel('Relative Importance')
plt.title("Decision Tree regressor")
plt.show()

# We just take the values from the feature_importances_ method and scale them so
# that they range between 0 and 100. Let's see what we will get for a decision
# tree-based regressor in the following output: So, the decision tree regressor says that the most important feature is RM.

# 4 Now we carry out a similar procedure for the AdaBoost model:
ABFImp= ab_regressor.feature_importances_ 
ABFImp= 100.0 * (ABFImp / max(ABFImp))
index_sorted = np.flipud(np.argsort(ABFImp))
pos = np.arange(index_sorted.shape[0]) + 0.5

# 5 To visualize the results, we will plot the bar graph:
plt.figure()
plt.bar(pos, ABFImp[index_sorted], align='center')
plt.xticks(pos, housing_data.feature_names[index_sorted])
plt.ylabel('Relative Importance')
plt.title("AdaBoost regressor")
plt.show()

# Let's take a look at what AdaBoost has to say in the following output:
# According to AdaBoost, the most important feature is LSTAT. In reality,
# if you build various regressors on this data,
# you will see that the most important feature is in fact LSTAT. This shows
# the advantage of using AdaBoost with a decision tree-based regressor.
