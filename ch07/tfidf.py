# 1 Create a new python file and import the following package
# Let's make the basic imports
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

# 2 Let's select a list of categories and name them
# using a dictionnary mapping. These categories are available
# as a part of the news groups dataset that we just imported
category_map = {'misc.forsale': 'Sales', 'rec.motorcycles': 'Motorcycles',
                'rec.sport.baseball': 'Baseball', 'sci.crypt': 'Cryptography',
                'sci.space': 'Space'}

# 3 Load the training data based on the categories that we
# just defined
training_data = fetch_20newsgroups(
    subset='train', categories=category_map.keys(), shuffle=True, random_state=7)

# 4 Import the feature extractor
# Feature extraction

# 5 Extract the features by using the training data
vectorizer = CountVectorizer()
X_train_termcounts = vectorizer.fit_transform(training_data.data)
print("Dimension of the training data: ", X_train_termcounts.shape)

# 6 We are now ready to train the classifier. We will use the
# multinomial Naive Bayes classifier
# Training a classifier

# 7 Define a couple of random input sentences:
input_data = input_data = ["The curveballs of right handed pitchers tend to curve to the left",
                           "Caesar cipher is an ancient form of encryption", "This two-wheeler is really good on slippery roads"]

# 8 Define the tfidf_transformer object and train it
# tf.idf transformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_termcounts)

# 9 Once we have the feature vectors, train the multinomial Naive Bayes classifier using this data:
# Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(X_train_tfidf, training_data.target)

# 10 Transform the input data using the word counts
X_input_termcounts = vectorizer.transform(input_data)

# 11 Transform the input data using the tfidf_transfmer module
X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)

# 12 Predict the output categories of these input sentences by using the trained classifier:
# Predict the output categories
predicted_categories = classifier.predict(X_input_tfidf)

# 13 Print the output, as follows:
# Print the outputs
for sentence, category in zip(input_data, predicted_categories):
    print('\nInput:', sentence, '\nPredicted category:',
          category_map[training_data.target_names[category]])

# 14 Let's run
