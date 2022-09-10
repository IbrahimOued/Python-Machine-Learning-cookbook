# 1 Let's make the basic imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

# 2 Import the data from the sklearn dataset
IrisData = load_iris()

# 3 Divide the data into input and target
X = IrisData.data
y = IrisData.target.reshape(-1, 1)
# For the target, the data was converted to a single column

# 4 Let's encode the class labels as One hot encode
encoder = OneHotEncoder(sparse=False)
YHE = encoder.fit_transform(y)

# 5 Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, YHE, test_size=.3)

# 6 Let's build the model
model = Sequential()

# 7 Three layers will be added: an input layer, a hidden layer, and an output layerrrrr
model.add(Dense(10, input_shape=(4, ), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 8 Let's compile the model
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# The following arguments are passed
# * optimize='SGD' => Stochastic gradient descent optimizer. Includes support for momentum, learning rate decay, and Nesterov momentum
# * loss='categorical_crossentropy': We have used the categorical_crossentropy argument here. When using categorical_crossentropy, your targets should be in categorical format (we have three classes; the target for each sample must be a three-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample).
# * metrics=['accuracy']: A metric is a function that is used to evaluate the
# performance of your model during training and testing.

# 9 Let's train the model
model.fit(X_train, y_train, verbose=2, batch_size=5, epochs=200)

# 10 Finally, test the model using unseen data
results = model.evaluate(X_test, y_test)
print('Final test set loss:', results[0])
print('Final test set accuracy:', results[1])


# 11 Now let's see what happens if we use a different optimizer. To do this, just change the optimizer
# parameter in the compile method as follows
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# The adam optimizer is an algorithm for the first-order, gradient-based optimization of stochastic objective functions,
# based on adaptive estimates of lower-order moments. 