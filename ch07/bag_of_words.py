# 1 Let's make the basic imports
import numpy as np
from nltk.corpus import brown
from chunking import splitter

# 2 Let's define the main function. Load the input data from the brown corpus
if __name__=='__main__':
    # Read the data from the brown corpus
    data=' '.join(brown.words()[:1000])

# 3 Divide the text data into five chunks:
# Number of words in each chunk
num_words = 2000

chunks = []
counter=0

text_chunks=splitter(data, num_words)

# 4 Create a dictionnary that is based on these text chunks:
for text in text_chunks:
    chunk={'index': counter, 'text': text}
    chunks.append(chunk)
    counter+=1

# 5 The next step is to extract a document term matrix. This is basically a matrix that counts
# the number of occurences of each wor in the document.
# We will use scikit-learn to do this because it has better provisions, compared to NLTK,
# for this particular task. Import the following package 
# Extract document term matrix 
from sklearn.feature_extraction.text import CountVectorizer 

# 6 Define the object and extract the document term matrix:
vectorizer = CountVectorizer(min_df=5, max_df=.95) 
doc_term_matrix = vectorizer.fit_transform([chunk['text'] for chunk in chunks]) 

# 7 Extract the vocabulary from the vectorizer object and print it:
vocab = np.array(vectorizer.get_feature_names()) 
print("Vocabulary:")
print(vocab)

# 8 Print the Document term matrix
print("Document term matrix:") 
chunk_names = ['Chunk-0', 'Chunk-1', 'Chunk-2', 'Chunk-3', 'Chunk-4'] 

# 9 To print it in a tabular form, you will need to format this, as follows:
formatted_row = '{:>12}' * (len(chunk_names) + 1) 
print('\n', formatted_row.format('Word', *chunk_names), '\n') 

# 10 Iterate through the words and print the number of times each word has occurred in different chunks:
for word, item in zip(vocab, doc_term_matrix.T): 
    # 'item' is a 'csr_matrix' data structure 
    output = [str(x) for x in item.data] 
    print(formatted_row.format(word, *output)) 

# 11 Let's run