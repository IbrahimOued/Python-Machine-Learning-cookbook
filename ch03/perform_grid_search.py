# 1 Let's make the basic imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import utilities

# 2 Then load the data
input_file = 'ch03/data_multivar.txt'
X, y = utilities.load_data(input_file)

# 3 We split the data into a train and test dataset:
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.25, random_state=5)

# 4 Now, we will use cross-validation here, which we covered in the previous recipes.
# Once you load the data and split it into training and testing datasets, add the following to the file
# Set the parameters by cross-validation
parameter_grid = {"C": [1, 10, 50, 600],
                  'kernel': ['linear', 'poly', 'rbf'],
                  "gamma": [0.01, 0.001],
                  'degree': [2, 3]}
# 5 Let's define the metrics that we want to use
metrics = ['precision']

# 6 Let's start the search for optimal hyperparameters for each of the metrics:
for metric in metrics:
    classifier = GridSearchCV(svm.SVC(C=1),
                              parameter_grid, cv=5, scoring=metric, return_train_score=True)

    classifier.fit(X_train, y_train)

# 7 Let's look at the score (confidence)
print("Scores across the parameter grid:")
GridSCVResults = pd.DataFrame(classifier.cv_results_)
for i in range(0, len(GridSCVResults)):
    print(GridSCVResults.params[i], '-->',
          round(GridSCVResults.mean_test_score[i], 3))

    # 8 Let's print the best parameter set:
    print("Highest scoring parameter set:", classifier.best_params_)

# 10 As we can see in the preceding output, it searches for all the
# optimal hyperparameters. In this case, the hyperparameters are
# the type of kernel, the C value, and gamma. It will try out
# various combinations of these parameters to find the best parameters.
# Let's test it out on the testing dataset:
y_true, y_pred = y_test, classifier.predict(X_test)
print("Full performance report:\n")
print(classification_report(y_true, y_pred))

# 11 Runnung the code will give the full report
# 12 We have previously said that there are different techniques for
# optimizing hyperparameters. We'll apply the RandomizedSearchCV method.
# To do this, just use the same data and change the classifier. To the code just seen, we add a further section:
# Perform a randomized search on hyper parameters
parameter_rand = {'C': [1, 10, 50, 600],
                  'kernel': ['linear', 'poly', 'rbf'],
                  'gamma': [0.01, 0.001],
                  'degree': [2, 3]}
metrics = ['precision']
for metric in metrics:
    print("#### Randomized Searching optimal hyperparameters for", metric)
    classifier = RandomizedSearchCV(svm.SVC(C=1),
                                    param_distributions=parameter_rand, n_iter=30,
                                    cv=5, return_train_score=True)
    classifier.fit(X_train, y_train)
    print("Scores across the parameter grid:")
    RandSCVResults = pd.DataFrame(classifier.cv_results_)
    for i in range(0, len(RandSCVResults)):
        print(RandSCVResults.params[i], '-->',
              round(RandSCVResults.mean_test_score[i]))
# 13 Let's run and check the terminal
# 14 Let's test it out on the testing dataset:print("Highest scoring parameter set:", classifier.best_params_)
y_true, y_pred = y_test, classifier.predict(X_test)
print("Full performance report:\n")
print(classification_report(y_true, y_pred))

# 15 And we'll have the full report in the terminal
