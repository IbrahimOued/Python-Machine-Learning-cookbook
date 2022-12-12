# 1 Create the file and make the basic imports
from numpy import vectorize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 2 Load the spam.csv file
df=pd.read_csv('ch07/spam.csv', sep=',', header=None, encoding='latin-1')

# 3 Let's extract the data for training and testing
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

# 4 We need to vectorize the text data contained in the DataFrame
vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train_raw)

# 5 We can now build the logistic regression model
classifier=LogisticRegression()
classifier.fit(X_train, y_train)

# 6 Define 2 SMS messages as test data
X_test = vectorizer.transform( ['Customer Loyalty Offer:The NEW Nokia6650 Mobile from ONLY å£10 at TXTAUCTION!', 'Hi Dear how long have we not heard.'] )

# 7 Finally, we will perforù a prediction by using the model
predictions = classifier.predict(X_test)
print(predictions)


# The following results will be returned:

# ['spam' 'ham']
# These indicate that the first SMS was identified as spam, while the second SMS was identified as ham.
