# 1 Let's make the basic imports
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

# 2 To import the MNIST dataset
(XTrain, YTrain), (XTest, YTest) = mnist.load_data()

print('XTrain shape = ', XTrain.shape)
print('XTest shape = ', XTest.shape)
print('YTrain shape = ', YTrain.shape)
print('YTest shape = ', YTest.shape)

# 3 After importing the dataset, we have printed the shape of the data

# 4 The 70,000 items in the database were divided into 60,000 items for
# training, and 10,000 items for testing. The data output is represented
# by integers in the range 0 to 9. Let's check it as follows:
print('YTrain values = ', np.unique(YTrain))
print('YTest values = ', np.unique(YTest))

# 5 The following results are printed:
# YTrain values =  [5 0 4 2 1 3 6 7 8 9]
# YTest values =  [5 0 4 2 1 3 6 7 8 9]

# 6 It may be useful to analyze the distribution of the two values
# in the available arrays. To start, we count the number of occurrences:
unique, counts = np.unique(YTrain, return_counts=True)
print('YTrain distribution = ', dict(zip(unique, counts)))
unique, counts = np.unique(YTest, return_counts=True)
print('YTrain distribution = ', dict(zip(unique, counts)))

# 7 The following results are returned:
# YTrain distribution = {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
# YTrain distribution = {0: 980, 1: 1135, 2: 1032, 3: 1010, 4: 982, 5: 892, 6: 958, 7: 1028, 8: 974, 9: 1009}

# 8 We can also see it in a graph, as follows:
plt.figure(1)
plt.subplot(121)
plt.hist(YTrain, alpha=0.8, ec='black')
plt.xlabel("Classes")
plt.ylabel("Number of occurrences")
plt.title("YTrain data")

plt.subplot(122)
plt.hist(YTest, alpha=0.8, ec='black')
plt.xlabel("Classes")
plt.ylabel("Number of occurrences")
plt.title("YTest data")
plt.show()

# 9 To compare the results obtained on both output datasets (YTrain and YTest), two histograms were
# traced and displayed side by side, as shown in the following output:

# 10 From the analysis of the previous output, we can see that in both datasets, the 10 digits
# are represented in the same proportions. In fact, the bars seem to have the same dimensions,
# even if the vertical axis has different ranges.

# 11 Now, we have to normalize all values between 0 and 1:
XTrain = XTrain.astype('float32') / 255
XTest = XTest.astype('float32') / 255

# 12 To reduce the dimensionality, we will flatten the 28 x 28 images into vectors of size 784:
XTrain = XTrain.reshape((len(XTrain), np.prod(XTrain.shape[1:])))
XTest = XTest.reshape((len(XTest), np.prod(XTest.shape[1:])))

# 13 Now, we will build the model using the Keras functional API. Let's start importing the libraries:

# 14 Then, we can build the Keras model
InputModel = Input(shape=(784,))
EncodedLayer = Dense(32, activation='relu')(InputModel)
DecodedLayer = Dense(784, activation='sigmoid')(EncodedLayer)
AutoencoderModel = Model(InputModel, DecodedLayer)
AutoencoderModel.summary()

# 15 So, we have to configure the model for training. To do this, we will use the compile method, as follows:
AutoencoderModel.compile(optimizer='adadelta', loss='binary_crossentropy')

# 16 At this point, we can train the model
history = AutoencoderModel.fit(XTrain, XTrain,
                               batch_size=256,
                               epochs=100,
                               shuffle=True,
                               validation_data=(XTest, XTest))

# 17 Our model is now ready, so we can use it to rebuild the handwritten digits automatically.
# To do this, we will use the predict() method:
DecodedDigits = AutoencoderModel.predict(XTest)

# 18 We have now finished; the model has been trained and will later be used to make predictions.
# So, we can just print the starting handwritten digits and those that were reconstructed from our model. Of course,
# we will do it only for some of the 60,000 digits contained in the dataset. In fact, we will
# limit ourselves to displaying the first five; we will also use the matplotlib library in this case:
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(XTest[i+10].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(DecodedDigits[i+10].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()