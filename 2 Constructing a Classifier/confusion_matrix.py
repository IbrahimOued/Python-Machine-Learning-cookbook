# 1 We will use the confusio_matrix.py file that
# we already provided, let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# We use some sample data here. We have 4 classes
# with values ranging from 0 to 3. We have predicted
# labels as well. We use the confusion_matrix method to
# extract the confusion matrix and plot it.

# 2 Let's go ahead and define this function
# show confusion matrix

def plot_confusion_matrix(confusion_matrix):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Paired)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# We use the imshow function to plot the confusion matrix. Everything else in
# the function is straightforward! We just set the title, color bar, ticks,
# and the labels using the relevant functions. The tick_marks argument range
# from 0 to 3 because we have 4 distinct labels in our dataset. The np.arange
# function gives us this numpy array.

# 3 Let's define the data(real and predicted) and then we will call the confusion_matrix function
y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
confusion_mat = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(confusion_mat)

# 4 run the code and see the output
# The diagonal colors are strong, and we want them to be strong.
# The black color indicates zero. There are a couple of gray squares in the non-diagonal spaces, which indicate misclassification. For example, when the real label is 0, the predicted label is 1, as we can see in the first row. In fact, all the misclassifications belong to class 1 in the sense that the second column contains 3 rows
# that are non-zero. It's easy to see this from the matrix.
