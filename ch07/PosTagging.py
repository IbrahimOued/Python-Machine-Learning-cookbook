# 1 Create the file and make the basic imports
import spacy

# 2 Load the en_core_web_sm model
nlp=spacy.load('en_core_web_sm')

# 3 Let's define an input text
text=nlp(u'We catched fish, and talked, and we took a swim now and then to keep off sleepiness')
# As a source, I used a passage based on the novel The Adventures of Huckleberry Finn by Mark Twain.

# 4 Finally, we will perform PoS tagging
for token in text:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

# 5 Let's run it