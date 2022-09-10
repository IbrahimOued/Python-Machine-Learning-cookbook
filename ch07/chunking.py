# 1 Let's make the basic imports
import numpy as np
# nltk.download('brown')
from nltk.corpus import  brown

# 2 Let's define a function to split the text into chunks.
# The first step is to divide the text based on spaces
# Split a text into chunks
def splitter(data, num_words):
    words = data.split(' ')
    output=[]

    # 3 Initialize a couple of required variables
    cur_count=0
    cur_words=[]

    # 4 Lets iterate through the words
    for word in words:
        cur_words.append(word)
        cur_count += 1

        # 5 Once you have hit the required number of words, reset the variables
        if cur_count == num_words:
            output.append(' '.join(cur_words))
            cur_words=[]
            cur_count= 0

    # 6 Append the chunks to the output variable, and return it
    output.append(' '.join(cur_words))
    return output

# 7 We can now define the main function. Load the data from brown corpus.
# We will use the first 10.000 rows
if __name__=='__main__': 
    # Read the data from the Brown corpus 
    data = ' '.join(brown.words()[:10000]) 

    # 8 Define the number of words in each chunk:
    # Number of words in each chunk  
    num_words = 1700

    # 9 Initialize a couple of relevant variables:
    chunks = [] 
    counter = 0 

    # 10 Call the splitter function on this text data and print the output:
    text_chunks = splitter(data, num_words) 
 
    print("Number of text chunks =", len(text_chunks)) 

# 11 Let's run