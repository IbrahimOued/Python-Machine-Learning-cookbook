# 1 Let's make the basic imports
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import metrics 
from sklearn.cluster import KMeans 

# 2 Let's load the data
input_file = ('ch04/data_perf.txt')

x = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        x.append(data)

data = np.array(x)

# 3 In order to determine the optimal number of
# clusters, let's iterate through a range of values and see where it peaks:
scores = [] 
range_values = np.arange(2, 10) 
 
for i in range_values: 
    # Train the model 
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10) 
    kmeans.fit(data) 
    score = metrics.silhouette_score(data, kmeans.labels_,  
                metric='euclidean', sample_size=len(data)) 
 
    print("Number of clusters =", i)
    print("Silhouette score =", score)
                     
    scores.append(score)

# 4 Now let's plot the graph to see where it peaked:
# Plot scores 
plt.figure() 
plt.bar(range_values, scores, width=0.6, color='k', align='center') 
plt.title('Silhouette score vs number of clusters') 
 
# Plot data 
plt.figure() 
plt.scatter(data[:,0], data[:,1], color='k', s=30, marker='o', facecolors='none') 
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1 
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1 
plt.title('Input data') 
plt.xlim(x_min, x_max) 
plt.ylim(y_min, y_max) 
plt.xticks(()) 
plt.yticks(()) 
 
plt.show()

# 5 Run this code, we'll se that 5 apprears as the ideal number of clusters
# As with these scores, the best configuration is five clusters.

# We can visually confirm that the data, in fact,
# has five clusters. We just took the example of a small
# dataset that contains five distinct clusters.
# This method becomes very useful when you are dealing with a
# huge dataset that contains high-dimensional data that cannot be visualized easily.