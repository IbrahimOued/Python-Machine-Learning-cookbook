# 1 Let's make the basic imports and load the data
import numpy as np 
from sklearn import preprocessing 
from sklearn.svm import SVC

input_file = 'ch03/building_event_binary.txt' 
# input_file = 'ch03/building_event_mutliclass.txt' 
 
# Reading the data 
X = [] 
count = 0 
with open(input_file, 'r') as f: 
    for line in f.readlines(): 
        data = line[:-1].split(',') 
        X.append([data[0]] + data[2:]) 
 
X = np.array(X) 

# 2 Let's convert the data into numerical form
# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, -1] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# 3 Let's train the SVM using the radial basis function, Platt scaling, and class balancing:
# Build SVM 
params = {'kernel': 'rbf', 'probability': True, 'class_weight': 'balanced'}  
classifier = SVC(**params, gamma='auto') 
classifier.fit(X, y) 

# 4 We are now ready to perform cross-validation:
from sklearn import model_selection
accuracy = model_selection.cross_val_score(classifier, 
        X, y, scoring='accuracy', cv=3)
print("Accuracy of the classifier: " + str(round(100*accuracy.mean(), 2)) + "%")

# 5 Let's test our SVM on a new datapoint:
# Testing encoding on single data instance
input_data = ['Tuesday', '12:30:00','21','23']
input_data_encoded = [-1] * len(input_data)
count = 0

for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count = count + 1 

input_data_encoded = np.array(input_data_encoded)

# Predict and print(output for a particular datapoint
output_class = classifier.predict([input_data_encoded])
print("Output class:", label_encoder[-1].inverse_transform(output_class)[0])

# 6 f you run this code, you will see the following output on your Terminal:
# Accuracy of the classifier: 93.95%
# Output class: noevent  

# If you use the building_event_multiclass.txt file as the input data file
# instead of building_event_binary.txt, you will see the following output on your Terminal:
# Accuracy of the classifier: 65.33%
# Output class: eventA