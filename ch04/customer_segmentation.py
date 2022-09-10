# 1 Let's make the basic imports
import csv 
import numpy as np 
from sklearn.cluster import MeanShift, estimate_bandwidth 
import matplotlib.pyplot as plt 

# 2 Let's load the input data
# Load data from input file 
input_file = 'ch04/wholesale.csv' 
file_reader = csv.reader(open(input_file, 'rt'), delimiter=',') 
X = [] 
for count, row in enumerate(file_reader): 
    if not count: 
        names = row[2:] 
        continue 
 
    X.append([float(x) for x in row[2:]]) 
 
# Input data as numpy array 
X = np.array(X)

# 3 Let's build a mean shift model
# Estimating the bandwidth  
bandwidth = estimate_bandwidth(X, quantile=0.8, n_samples=len(X)) 
 
# Compute clustering with MeanShift 
meanshift_estimator = MeanShift(bandwidth=bandwidth, bin_seeding=True) 
meanshift_estimator.fit(X) 
labels = meanshift_estimator.labels_ 
centroids = meanshift_estimator.cluster_centers_ 
num_clusters = len(np.unique(labels)) 
 
print("Number of clusters in input data =", num_clusters)

# 4 Let's print the centroids of clusters that we obtained, as follows:
print('\t'.join([name[:3] for name in names]))
for centroid in centroids:
    print('\t'.join([str(int(x)) for x in centroid]))

# 5 Let's visualize a couple of features to get a sense of the output:
# Visualizing data 
 
centroids_milk_groceries = centroids[:, 1:3] 
 
# Plot the nodes using the coordinates of our centroids_milk_groceries 
plt.figure() 
plt.scatter(centroids_milk_groceries[:,0], centroids_milk_groceries[:,1],  
        s=100, edgecolors='k', facecolors='none') 
 
offset = 0.2 
plt.xlim(centroids_milk_groceries[:,0].min() - offset * centroids_milk_groceries[:,0].ptp(), 
        centroids_milk_groceries[:,0].max() + offset * centroids_milk_groceries[:,0].ptp(),) 
plt.ylim(centroids_milk_groceries[:,1].min() - offset * centroids_milk_groceries[:,1].ptp(), 
        centroids_milk_groceries[:,1].max() + offset * centroids_milk_groceries[:,1].ptp()) 
 
plt.title('Centroids of clusters for milk and groceries') 
plt.show()

# 6 Let's run the code
# In this output, the eight centroids of the identified clusters are clearly represented.