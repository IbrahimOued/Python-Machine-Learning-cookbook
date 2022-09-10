# 1 Let's make the basic imports
from sklearn.datasets import samples_generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

# 2 Let's generate some sample data to play with, as follows
# generate sample data
X, y = samples_generator.make_classification(
    n_informative=4, n_features=20, n_redundant=0, random_state=5)
# This line generated 20 dimensional feature vectors because this is the default value.
# You can change it using the n_features parameter in the previous line.

# 3 Our first step of the pipeline is to select the k best features before the
# datapoint is used further. In this case, let's set k to 10:
# Feature selector  
selector_k_best = SelectKBest(f_regression, k=10)

# 4 The next step is to use a random forest classifier method to classify the data:
# Random forest classifier 
classifier = RandomForestClassifier(n_estimators=50, max_depth=4)

# 5 We are now ready to build the pipeline. The Pipeline() method allows us to
# use predefined objects to build the pipeline:
# Build the machine learning pipeline 
pipeline_classifier = Pipeline([('selector', selector_k_best), ('rf', classifier)])
# We can also assign names to the blocks in our pipeline. In the preceding line,
# we'll assign the selector name to our feature selector, and rf to our random
# forest classifier. You are free to use any other random names here!

# 6 We can also update these parameters as we go along. We can set the parameters
# using the names that we assigned in the previous step. For example, if we want
# to set k to 6 in the feature selector and set n_estimators to 25 in the random
# forest classifier, we can do so as demonstrated in the following code. Note that
# these are the variable names given in the previous step:
pipeline_classifier.set_params(selector__k=6, rf__n_estimators=25) 

# 7 Let's go ahead and train the classifier:
# Training the classifier 
pipeline_classifier.fit(X, y)

# 8 Let's now predict the output for the training data, as follows:
# Predict the output
prediction = pipeline_classifier.predict(X)
print("Predictions:\n", prediction)

# 9 Now, let's estimate the performance of this classifier, as follows
# Print score 
print("Score:", pipeline_classifier.score(X, y))

# 10 We can also see which features will get selected, so let's go ahead and print them:
# Print the selected features chosen by the selector
features_status = pipeline_classifier.named_steps['selector'].get_support()
selected_features = []
for count, item in enumerate(features_status):
    if item:
        selected_features.append(count)

print("Selected features (0-indexed):", ', '.join([str(x) for x in selected_features]))