# 1 Let's make the basic imports
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

# 2 Define a function to extract the features


def extract_features():
    return dict([(word, True) for word in word_list])


# 3 We need training data for this, so we will use the movie reviews in NLTK
if __name__ == '__main__':
    # Load positive and negative reviews
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')

# 4 Let's separate them into positive and negative reviews
features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                      'Positive') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                      'Negative') for f in negative_fileids]
# 5 Divide the data into train and test datasets
# Split the data into train and test (80/20)
threshold_factor=.8
threshold_positive=int(threshold_factor*len(features_positive))
threshold_negative=int(threshold_factor*len(features_negative))

# 6 Extract the features
features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative] 
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]   
print("Number of training datapoints:", len(features_train))
print("Number of test datapoints:", len(features_test))

# 7 We will use a NaiveBayesClassifier. Define the object and train it
# Train a Naive Bayes classifier 
classifier = NaiveBayesClassifier.train(features_train) 
print("Accuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))

# 8 The classifier object contains the most informative words that it obtained during 
# analysis. These words basically have a strong say in what's classified as a positive or a 
# negative review. Let's print them out:
print("Top 10 most informative words:")
for item in classifier.most_informative_features()[:10]:
    print(item[0])

# 9 Create a couple of random input sentences:
# Sample input reviews 
input_reviews = [ 
    "It is an amazing movie",  
    "This is a dull movie. I would never recommend it to anyone.", 
    "The cinematography is pretty great in this movie",  
    "The direction was terrible and the story was all over the place" ,
    "Please help me with the cable I'm trying to connect on the laptop"
] 

# 10 Run the classifier on those input sentences and obtain the predictions:
print("Predictions:") 
for review in input_reviews: 
    print("Review:", review) 
    probdist = classifier.prob_classify(extract_features(review.split())) 
    pred_sentiment = probdist.max() 

# 11 Print the output:
print("Predicted sentiment:", pred_sentiment) 
print("Probability:", round(probdist.prob(pred_sentiment), 2))

# 12 Let's run it
# 13 The next item is a list of the most informative words:
# 14 The last item is the list of predictions, which are based on the input sentences:
