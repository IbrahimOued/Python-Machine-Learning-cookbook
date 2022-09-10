# 1 Let's make the usual imports
from sklearn.datasets import fetch_20newsgroups

# This dataset is contained in the sklearn.datasets library; 
# in this way, it will be very easy for us to recover the data.
# As anticipated, the dataset contains posts related to
# 20 newsgroups. We will limit our analysis to only the following two newsgroups:

NewsClass = ['rec.sport.baseball', 'rec.sport.hockey']

# 2 Download the data:
DataTrain = fetch_20newsgroups(subset='train',categories= NewsClass, shuffle=True, random_state=42)

# 3 The data has 2 attributes: data and target. 
# Obviously, data represents the input and target is the output.
# Let's check which newsgroups have been selected:

print(DataTrain.target_names)

# 4 Let's check the shape:
print(len(DataTrain.data))
print(len(DataTrain.target))

# 5 To extract features from texts, we will use the CountVectorizer() function as follows:
from sklearn.feature_extraction.text import CountVectorizer

CountVect = CountVectorizer()
XTrainCounts = CountVect.fit_transform(DataTrain.data)
print(XTrainCounts.shape)

# In this way, we have made a count of the occurrences of words.

# 6 Now let's divide the number of occurrences of each word in a document by the total number of words in the document:
from sklearn.feature_extraction.text import TfidfTransformer

TfTransformer = TfidfTransformer(use_idf=False).fit(XTrainCounts)
XTrainNew = TfTransformer.transform(XTrainCounts)
TfidfTransformer = TfidfTransformer()
XTrainNewidf = TfidfTransformer.fit_transform(XTrainCounts)

# 7 Now we can build the classifier:
from sklearn.naive_bayes import MultinomialNB

NBMultiClassifier = MultinomialNB().fit(XTrainNewidf, DataTrain.target)

# 8 Finally, we will compute the accuracy of the classifier:
NewsClassPred = NBMultiClassifier.predict(XTrainNewidf)

accuracy = 100.0 * (DataTrain.target == NewsClassPred).sum() / XTrainNewidf.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")
