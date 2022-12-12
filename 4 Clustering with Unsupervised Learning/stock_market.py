# 1 Let's make the basic imports
import json
import sys
import pandas as pd

import numpy as np
from sklearn import covariance, cluster

# 2 We need a file that contains all the symbols and the
# associated names. This information is located in the
# symbol_map.json file provided to you. Let's load this, as follows:
# Input symbol file 
symbol_file = 'ch04/symbol_map.json' 

# 3 Let's read the data from the symbol_map.json file:
# Load the symbol map 
with open(symbol_file, 'r') as f: 
    symbol_dict = json.loads(f.read()) 
 
symbols, names = np.array(list(symbol_dict.items())).T

# 4 Now let's load the data. We will use an Excel file
# (stock_market_data.xlsx); this is a multisheet file, one for each symbol:
quotes = []

excel_file = 'ch04/stock_market_data.xlsx'

for symbol in symbols:
    print('Quote history for %r' % symbol, file=sys.stderr)
    quotes.append(pd.read_excel(excel_file, symbol))

# 5 As we need some feature points for analysis, we will use the difference
# between the opening and closing quotes every day to analyze the data:
# Extract opening and closing quotes 
opening_quotes = np.array([quote.open for quote in quotes]).astype(np.float) 
closing_quotes = np.array([quote.close for quote in quotes]).astype(np.float) 
 
# The daily fluctuations of the quotes  
delta_quotes = closing_quotes - opening_quotes

# 7 Let's build a graph model:
# Build a graph model from the correlations 
edge_model = covariance.GraphicalLassoCV(cv=3)

# 8 We need to standardize the data before we use it:
# Standardize the data  
X = delta_quotes.copy().T 
X /= X.std(axis=0) 

# 9 Now let's train the model using this data
# Train the model 
with np.errstate(invalid='ignore'): 
    edge_model.fit(X)

# 10 We are now ready to build the clustering model, as follows:
# Build clustering model using affinity propagation 
_, labels = cluster.affinity_propagation(edge_model.covariance_) 
num_labels = labels.max() 
 
# Print the results of clustering 
for i in range(num_labels + 1): 
    print("Cluster", i+1, "-->", ', '.join(names[labels == i]))

# 11 Let's run the code
# Eight clusters are identified. From an initial analysis,
# we can see that the grouped companies seem to treat the
# same products: IT, banks, engineering, detergents, and computers.