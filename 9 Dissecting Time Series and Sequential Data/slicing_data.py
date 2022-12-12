# 1 Let's make the basic imports
import numpy as np
import matplotlib.pyplot as plt
from convert_to_timeseries import convert_data_to_timeseries

# 2 We will use the same text file taht we used in the previous recipe
# Input file containing data
input_file = 'ch09/data_timeseries.txt'

# 3 We will extract only 3rd column
# load data
colum_num = 2
data_timeseries = convert_data_to_timeseries(input_file, colum_num)

# 4 Let's assume that we want to extract the data between the given
# start and end years. Let's define these
# Plot within a certain year range
start = '2000'
end = '2015'

# 5 Plot the data between the given year range
plt.figure()
data_timeseries[start:end].plot()
plt.title('data from ' + start + ' to ' + end)

# 6 We can also slice the data based on a certain range of months
# plot within a certain range of dates
start = '2008-01'
end = '2008-12'

# 7 Plot the data
plt.figure()
data_timeseries[start:end].plot()
plt.title("Data from " + start + " to " + end)
plt.show() 