# 1 Let's make the basic imports
import nltk
nltk.download('punkt')

# 2 Let's define some sample text for analysis
text = "Are you curious about tokenization? Let's see how it works! We need to analyze a couple of sentences with punctuations to see it in action."

# 3 Let's start with sentence tokenization. NLTK provices a sentence tokenizer, so let's import it
# sentence tokenization
from nltk.tokenize import sent_tokenize

# 4 Run the sentence tokenizer on the input text and extract the tokens
sent_tokenize_list=sent_tokenize(text)

# 5 Print the list of sentences to see whether it works
print("Sentence tokenize: ")
print(sent_tokenize_list)

# 6 Word tokenization is very commonly used in NLP. NLTK comes with a couple of
# different word tokenizers. Let's start with the basic word tokenizer:
# create a new word tokenizer
from nltk.tokenize import word_tokenize

print('Word tokenize:')
print(word_tokenize(text))

# 7 If you want to split this punctuation into separate tokens, then you will need to use the WordPunct tokenizer:
# Create a new WordPunct tokenizer 
from nltk.tokenize import WordPunctTokenizer 
 
word_punct_tokenizer = WordPunctTokenizer() 
print("Word punct tokenizer:")
print(word_punct_tokenizer.tokenize(text)) 

# 8 Run the code