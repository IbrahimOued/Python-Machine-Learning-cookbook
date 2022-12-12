# 1 Let's make the basic imports
import nltk
nltk.download('names')
import random
from nltk.corpus import names
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy

# 2 We need to define a function to extract features from input words:
# Extract features from the input word 
def gender_features(word, num_letters=2):
    return {'feature': word[-num_letters:].lower()} 

# 3 Let's define the main function. We need some labeled training data:
if __name__=='__main__': 
    # Extract labeled names 
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] + 
            [(name, 'female') for name in names.words('female.txt')]) 

 # 4 Seed the random number generator and shuffle the training data
    random.seed(7)
    random.shuffle(labeled_names)

# 5 Define some input names to play with
input_names=['Eric', 'Moussa', 'Ibrahim', 'Abby', 'Sam']

# 6 As we don't know how many ending characters we need to consider, we will
# sweep the parameter space from 1 to 5. Each time, we will extract the features, as follows
# Sweeping the parameter space 
for i in range(1, 5): 
    print('\nNumber of letters:', i) 
    featuresets = [(gender_features(n, i), gender) for (n, gender) in labeled_names] 

# 7 Divide this into train and test datasets:
train_set, test_set=featuresets[500:], featuresets[:500]

# 8 We will use the Naive Bayes classifier to do this
classifier = NaiveBayesClassifier.train(train_set)

# 9 Evaluate the classifier model for each value in the parameter space:
# Print classifier accuracy 
print('Accuracy ==>', str(100 * nltk_accuracy(classifier, test_set)) + str('%')) 
 
# Predict outputs for new inputs 
for name in input_names: 
    print(name, '==>', classifier.classify(gender_features(name, i))) 

# 10 Let's run it