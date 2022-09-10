# 1 Let's make the basic imports
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# 2 Let's define a few words to play with, as follows:
words=['table', 'probably', 'wolves', 'playing', 'is',  
        'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision'] 

# 3 We'll define a list of stemmers that we want to use
# Compare different stemmers
stemmers=['PORTER', 'LANCASTER', 'SNOWBALL']

# 4 Initialize the required objects for all three stemmers
stemmer_porter = PorterStemmer()
stemmer_lancaster = LancasterStemmer()
stemmer_snowball=SnowballStemmer('english')

# 5 In order to print the output data in a neat tabular form,
# we need to format it in the correct way
formatted_row='{:>16}' * (len(stemmers) + 1) 
print('\n', formatted_row.format('WORD', *stemmers), '\n')

# 6 Let's iterate through the list of words and stem them by using
# the three stemmers
for word in words:
    stemmed_words = [stemmer_porter.stem(word), stemmer_lancaster.stem(word), stemmer_snowball.stem(word)]
    print(formatted_row.format(word, *stemmed_words))

# Let's run it