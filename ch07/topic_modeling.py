# 1 Create the file and make the basic imports
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
from nltk.corpus import stopwords

# 2 Define a function to load the input data. We ill use the data_topic_modeling.txt text
# file that has already been provided to you
# load input data
def load_data(input_file):
    data=[]
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1])
    return data

# 3 Let's define a class to preprocess the text. This preprocessor will take
# care of creating the required objects and extracting the relevant features
# from the input text
# class to preprocess text
class Preprocessor(object):
    # Initialize various operators
    def __init__(self) -> None:
        # create a regular expression tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

        # 4 We need a list of stop words so that we can exclude them from analysis. These
        # are common words, such as in, the, is and so on
        # get the list of stop words
        self.stop_words_english=stopwords.words('english')

        # 5 Define the SnowballStemmer module
        # Create a Snowball stemmer
        self.stemmer = SnowballStemmer('english')

    # 6 Define a processor function that takes care of tokenization, stop word removal, and stemming: # Tokenizing, stop word removal, and stemming 
    def process(self, input_text): 
        # Tokenize the string 
        tokens = self.tokenizer.tokenize(input_text.lower())

        # 7 Remove the stopwords
            # Remove the stop words  
        tokens_stopwords = [x for x in tokens if not x in self.stop_words_english] 

        # 8 Performing stemmng on the tokens
        # Perform stemming on the tokens
        tokens_stemmed=[self.stemmer.stem(x) for x in tokens_stopwords]

        # 9 Return the processed tokens
        return tokens_stemmed

# 10 We are now ready to define the main function. Load the input data
# form the text file
if __name__ == '__main__':
    # File containing lineswise input data
    input_file='ch07/data_topic_modeling.txt'
    # Load data
    data=load_data(input_file)

    # 11 Define an object that is based on the class that we defined:
    # Create a preprocessor object
    preprocessor=Preprocessor()

    # 12 We need to process the text in the file and extract the processed tokens
    # Create a list for processed documents
    processed_tokens=[preprocessor.process(x) for x in data]

    # 13 Create a dictionnary that is based on tokenized documents so that it can
    # be used for topic modeling
    # Create a dictionnary based on the tokenized documents
    dict_tokens=corpora.Dictionary(processed_tokens)

    # 14 We need to create a document term matrix usinf the processed tokens, as follows
    corpus=[dict_tokens.doc2bow(text) for text in processed_tokens]

    # 15 Let's suppose that we know that the text can be divided into 2 topics. We will use a
    # technique called Latent Dirichlet Allocation (LDA) for topic modelling. Define the required
    # parameters and initialize the LdaModel object
    # Generate the LDA model based on the corpus we just created
    num_topics=2
    num_words=4
    ldamodel=models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dict_tokens, passes=25)

    # 16 Let's suppose that we Once this has identified the two topics, we can see how it's separating these
    # two topics by looking at the most contributed words:
    print("Most contributing words to the topics:")
    for item in ldamodel.print_topics(num_topics=num_topics, num_words=num_words):
        print("Topic", item[0], "==>", item[1])

    # 17 Let's run it