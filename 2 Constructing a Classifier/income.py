# 1 We will use the Naive Bayes classifier to achieve this. Let's
# make the usual imports
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# 2 Let's load the dataset
input_file = 'ch02/adult.data.txt'
# reading the data
X = []
y = []
count_lessthan50k = 0
count_morethan50k = 0
num_images_threshold = 1000

# 3 We will use 20,000 datapoints from the datasets-10,000 for each
# class to avoid class imbalance. During training, if you use many datapoints
# that belong to a single class, the classifier tends to get biaised toward that ckass.
# Therefore, it's better to use the same number of datapoints for each class

with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_lessthan50k < num_images_threshold:
            X.append(data)
            count_lessthan50k = count_lessthan50k + 1
        elif data[-1] == '>=50K' and count_morethan50k < num_images_threshold:
            X.append(data)
            count_morethan50k = count_morethan50k + 1
        if count_lessthan50k >= num_images_threshold and count_morethan50k >= num_images_threshold:
            break
X = np.array(X)
# It's a comma-separated file again. We just loaded the data in the X variable just as before.

# 4 We need o convert string attributes to numerical data while leaving out the original numerical data:# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# The isdigit() function helps us to identify numerical data. We converted
# string data to numerical data and stored all the label encoders in a list
# so that we can use it when we want to classify unknown data.

# 5 Let's train the classifier
# Build a classifier
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)

# 6 Let's split the data into training and testing to extract performance metrics:# Cross validation
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb.predict(X_test)

# 7 Let's extract performance metrics:
# compute F1 score of the classifier
f1 = model_selection.cross_val_score(classifier_gaussiannb,
        X, y, scoring='f1_weighted', cv=5)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

# 8 Let's see how to classify a single datapoint. We need to convert the
# datapoint into something that our classifier can understand:
# Testing encoding on single data instance
input_data = ['39', 'State-gov', '77516', 'Bachelors', '13', 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male', '2174', '0', '40', 'United-States']
count = 0
input_data_encoded = [-1] * len(input_data)
for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int([input_data[i]])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count = count + 1 
input_data_encoded = np.array(input_data_encoded)

# 9 We are now ready to classify it:
# Predict and print output for a particular datapoint
output_class = classifier_gaussiannb.predict([input_data_encoded])
print(label_encoder[-1].inverse_transform(output_class)[0])
# Just as before, we use the predict method to get the output class and 
# the inverse_transform method to convert this label back to its original
# form to print it out on the Terminal. The following result is returned: <=50k

