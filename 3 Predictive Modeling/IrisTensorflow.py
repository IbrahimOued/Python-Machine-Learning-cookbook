# 1 We start with the basic imports
from sklearn import datasets
from sklearn import model_selection
import tensorflow as tf

# 2 Load the dataset
iris = datasets.load_iris()

# 3 Load and split the features and classes
X_train, y_train, X_test, y_test = model_selection.train_test_split(
    iris.data, iris.target, train_size=.7, random_state=1)

# 4 Now we will build a simple neural network with one hidden layer and 10 nodes:
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
classifier_tf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                               hidden_units=[10],
                                               n_classes=3)

# 5 Then we fit the network:
classifier_tf.fit(X_train, y_train, steps=5000)

# 6 We will then make the predictions:
predictions = list(classifier_tf.predict(X_test, as_iterable=True))

# 7 Finally, we will calculate the accuracy metric of the model:
n_items = y_test.size
accuracy = (y_test == predictions).sum() / n_items
print("Accuracy :", accuracy)