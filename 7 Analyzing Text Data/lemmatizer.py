# 1 Make the basic imports
import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# 2 Let's define the same set of words that we used during stemming
words = ['table', 'probably', 'wolves', 'playing', 'is', 'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision'] 

# 3 We will compare two lemmatizers: the NOUN and VERB lemmatizers. Let's list them:
# Compare different lemmatizers 
lemmatizers = ['NOUN LEMMATIZER', 'VERB LEMMATIZER'] 

# 4 Create the object based on the WordNet lemmatizer
lemmatizer_wordnet=WordNetLemmatizer()

# 5 In order to print the output in a tabular form, we need to format it in the right way
formatted_row = '{:>24}' * (len(lemmatizers) + 1) 
print('\n', formatted_row.format('WORD', *lemmatizers), '\n') 

# 6 Iterate through the words and lemmatize them
for word in words:
    lemmatized_words=[lemmatizer_wordnet.lemmatize(word, pos='n'), lemmatizer_wordnet.lemmatize(word, pos='v')]
    print(formatted_row.format(word, *lemmatized_words)) 

# 7 Let's run it