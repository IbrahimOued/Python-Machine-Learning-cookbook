# 1 Let's make the basic imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2 Let's define a function that reads an input file and converts sequential
# observations into time-indexed data
def convert_data_to_timeseries(input_file, column, verbose=False):
    # We will use a text file consisting of four columns.
    # The first column denotes the year,
    # the second column denotes the month,
    # and the third and fourth columns denote data. Let's load this into a NumPy array
    # Load the input file 
    data = np.loadtxt(input_file, delimiter=',')

    # 4 As this is arranged chronologically, the first row contains
    # the start date and the last row contains the end date. Let's
    # extract the start and end dates of this dataset
    # Extract the start and end dates 
    start_date = str(int(data[0,0])) + '-' + str(int(data[0,1])) 
    end_date = str(int(data[-1,0] + 1)) + '-' + str(int(data[-1,1] % 12 + 1))
    # 5 There is also a verbose mode for this function. So, if this is
    # set to true, it will print a few things. Let's print out the start and end dates:
    if verbose: 
        print("Start date =", start_date)
        print("End date =", end_date)

    # 6 Let's create a pandas variable, which contains the date sequence with monthly intervals
    # Create a date sequence with monthly intervals 
    dates = pd.date_range(start_date, end_date, freq='M')

    # 7 Our next step is to convert the given column into time series data.
    # You can access this data using the month and the year (as opposed to the index):
    # Convert the data into time series data 
    data_timeseries = pd.Series(data[:,column], index=dates) 

    # 8 Use the verbose mode to print out the first 10 elements
    if verbose: 
        print("Time series data:\n", data_timeseries[:10])
    # 9 Return the time-indexed variable, as follows:
    return data_timeseries 

# 10 Define the main function, as follows:
if __name__=='__main__': 
    # We will use the data_timeseries.txt file that is already provided to you:
    # Input file containing data 
    input_file = 'ch09/data_timeseries.txt' 

    # 12 Load the third column from this text file and convert it into time series data:
    # Load input data 
    column_num = 2 
    data_timeseries = convert_data_to_timeseries(input_file, column_num)

    # 13 The pandas library provides a nice plotting function that you can run directly on the variable:
     # Plot the time series data 
    data_timeseries.plot() 
    plt.title('Input data') 
 
    plt.show() 