# 1 Let's make the basic imports
# and load the data
# SVM regressor to estimate traffic 
 
import numpy as np 
from sklearn import preprocessing 
from sklearn.svm import SVR 
 
input_file = 'ch03/traffic_data.txt' 
 
# Reading the data 
X = [] 
count = 0 
with open(input_file, 'r') as f: 
    for line in f.readlines(): 
        data = line[:-1].split(',') 
        X.append(data) 
 
X = np.array(X)

# 2 Let's encode this data:
label_encoder = []  
X_encoded = np.empty(X.shape) 
for i,item in enumerate(X[0]): 
    if item.isdigit(): 
        X_encoded[:, i] = X[:, i] 
    else: 
        label_encoder.append(preprocessing.LabelEncoder()) 
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i]) 
 
X = X_encoded[:, :-1].astype(int) 
y = X_encoded[:, -1].astype(int) 

# 3 Let's build and train the SVM regressor using the radial basis function:
# Build SVR 
params = {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.2}  
regressor = SVR(**params) 
regressor.fit(X, y)
# In the preceding lines, the C parameter specifies the penalty for misclassification
# and epsilon specifies the limit within which no penalty is applied.

# 4 Let's perform cross-validation to check the performance of the regressor:
# Cross validation
import sklearn.metrics as sm

y_pred = regressor.predict(X)
print("Mean absolute error =", round(sm.mean_absolute_error(y, y_pred), 2))

# 5 Let's test it on a datapoint:
# Testing encoding on single data instance
input_data = ['Tuesday', '13:35', 'San Francisco', 'yes']
input_data_encoded = [-1] * len(input_data)
count = 0
for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count = count + 1 

input_data_encoded = np.array(input_data_encoded)

# Predict and print output for a particular datapoint
print("Predicted traffic:", int(regressor.predict([input_data_encoded])[0]))

# 6 Let's run and see the results on the terminal
