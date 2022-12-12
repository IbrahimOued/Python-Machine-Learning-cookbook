# 1 Create the file and make the basic imports
import gensim
from nltk.corpus import abc

# 2 Build a model based on the Word2Vec methodology
model=gensim.models.Word2Vec(abc.sents())

# 3 Let's extract the vocabulary from the data and put it into a list
X=model.wv.index_to_key

# 4 Now, we will find similarities with the word 'science'
data=model.wv.most_similar('science')

# 5 Finally, we will print the data
print(data)