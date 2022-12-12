# 1 Let's make the basic imports
import pandas as pd
import matplotlib.pyplot as plt
from convert_to_timeseries import convert_data_to_timeseries

# 2 We will use the same text file taht we used in the previous recipe
# Input file containing data
input_file = 'ch09/data_timeseries.txt'

# 3 We will use both the 3rd and 4th column in this txt file(remmeber, pyhton
# lists the data starting from position 0, so the 3rd column is 2, and the 4th will be 3
# in position)
# Load data
data1 = convert_data_to_timeseries(input_file, 2)
data2 = convert_data_to_timeseries(input_file, 3)

# 4 Convert the data into a pandas dataframe
dataframe = pd.DataFrame({'first': data1, 'second': data2})

# 5 Plot the data in the given year range
# Plot data
dataframe['1952':'1955'].plot()
plt.title('data overlapped on to of each year')

# 6 Let's assume that we want to plot the difference
# between the 2 columns that we just loaded on the fiven year range
# We can do this using the following lines
# Plot the difference 
plt.figure() 
difference = dataframe['1952':'1955']['first'] - dataframe['1952':'1955']['second'] 
difference.plot() 
plt.title('Difference (first - second)')

# 7 If we want to filter the data based on different conditions for the first and second
# columns, we can just specify these conditions and plot this:
# When 'first' is greater than a certain threshold 
# and 'second' is smaller than a certain threshold 
dataframe[(dataframe['first'] > 60) & (dataframe['second'] < 20)].plot(style='o') 
plt.title('first > 60 and second < 20')
plt.show()