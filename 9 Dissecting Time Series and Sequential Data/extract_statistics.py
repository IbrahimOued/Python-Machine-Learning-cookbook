# 1 Let's make the basic imports
import pandas as pd
import matplotlib.pyplot as plt
from convert_to_timeseries import convert_data_to_timeseries

# 2 We will use the same text file taht we used in the previous recipe
# Input file containing data
input_file = 'ch09/data_timeseries.txt'

# 3 Load both the data columns (3rd and 4th)
# Load data
data1 = convert_data_to_timeseries(input_file, 2)
data2 = convert_data_to_timeseries(input_file, 3)

# 4 Create a pandas data structure to hold this data. This dataframe is like
# a dictionnary that has keys and values
dataframe = pd.DataFrame({'first': data1, 'second': data2})

# 5 Let's start extracting some stats now. To extract the maximum and minimum values
# Print max and min
print('Maximum \n', dataframe.max())
print('Minimum \n', dataframe.min())

# 6 To print the mean values of your data or just the row-wise mean
# Print mean 
print('Mean:\n', dataframe.mean())
print('Mean row-wise:\n', dataframe.mean(1)[:10])

# 7 The rolling mean is an important statistic that's used a lot in
# time series processing. One of the most famous applications is
# smoothing a signal to remove noise. Rolling mean refers to
# computing the mean of a signal in a window that keeps sliding on the time scale.
# Let's consider a window size of 24 and plot this, as follows:
# Plot rolling mean
DFMean = dataframe.rolling(window=24).mean()
plt.plot(DFMean)

# 8 Correlation coefficients are useful in understanding the natire of the data
# Print correlation coefficients
print('COrrelation coefficients \n', dataframe.corr())

# 9 Let's plot this using a window size of 60
# Plot rolling correlation 
plt.figure()
DFCorr= dataframe.rolling(window=60).corr(pairwise=False)
plt.plot(DFCorr)
plt.show()
# The second output will indicate the rolling correlation (the following output
# is the result of a zoomed rectangle
# operation that was performed in the matplotlib window):

# 10 In the upper half of the Terminal, you will the see max, min, and mean
# values printed, as shown in the following output:

# 11 In the lower half of the terminal, you will see the row-wise mean stats and
# correlation coefficients printed, as shown in the following output:
