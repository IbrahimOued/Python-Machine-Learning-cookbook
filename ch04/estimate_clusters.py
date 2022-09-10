# 1 Let's make the basic imports
from itertools import cycle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

# 2 Load the input data
#  Load data
input_file = ('ch04/data_perf.txt')
x = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        x.append(data)

X = np.array(x)

# 3 We need to find the best parameter, so let's initialize a few variables:
# Find the best epsilon
eps_grid = np.linspace(0.3, 1.2, num=10)
silhouette_scores = []
eps_best = eps_grid[0]
silhouette_score_max = -1
model_best = None
labels_best = None

# 4 Let's sweep the parameter space:
for eps in eps_grid:
    # Train DBSCAN clustering model
    model = DBSCAN(eps=eps, min_samples=5).fit(X)

    # Extract labels
    labels = model.labels_

    # 5 For each iteration, we need to extract the performance metric:
    # Extract performance metric
    silhouette_score = round(metrics.silhouette_score(X, labels), 4)
    silhouette_scores.append(silhouette_score)

    print("Epsilon:", eps, " --> silhouette score:", silhouette_score)
    # 6 We need to store the best score and its associated epsilon value:
    if silhouette_score > silhouette_score_max:
        silhouette_score_max = silhouette_score
        eps_best = eps
        model_best = model 
        labels_best = labels 

# 7 Let's now plot the bar graph
# Plot silhouette scores vs epsilon 
plt.figure() 
plt.bar(eps_grid, silhouette_scores, width=0.05, color='k', align='center') 
plt.title('Silhouette score vs epsilon') 

# Best params 
print("Best epsilon =", eps_best)

# 8 Let's store the best models and labels:
# Associated model and labels for best epsilon 
model = model_best  
labels = labels_best

# 9 Some datapoints may remain unassigned. We need to identify them, as follows:
# Check for unassigned datapoints in the labels 
offset = 0 
if -1 in labels: 
    offset = 1 
# 10 Extract the number of clusters, as follows:
# Number of clusters in the data  
num_clusters = len(set(labels)) - offset  
 
print("Estimated number of clusters =", num_clusters)

# 11 We need to extract all the core samples, as follows:
# Extracts the core samples from the trained model 
mask_core = np.zeros(labels.shape, dtype=np.bool) 
mask_core[model.core_sample_indices_] = True 

# 12 Let's visualize the resultant clusters. We will start
# by extracting the set of unique labels and specifying different markers:
# Plot resultant clusters  
plt.figure() 
labels_uniq = set(labels) 
markers = cycle('vo^s<>') 

# 13 Now let's iterate through the clusters and plot the datapoints using different markers:
for cur_label, marker in zip(labels_uniq, markers): 
    # Use black dots for unassigned datapoints 
    if cur_label == -1: 
        marker = '.' 
 
    # Create mask for the current label 
    cur_mask = (labels == cur_label) 
 
    cur_data = X[cur_mask & mask_core] 
    plt.scatter(cur_data[:, 0], cur_data[:, 1], marker=marker, 
             edgecolors='black', s=96, facecolors='none') 
    cur_data = X[cur_mask & ~mask_core] 
    plt.scatter(cur_data[:, 0], cur_data[:, 1], marker=marker, 
             edgecolors='black', s=32) 
plt.title('Data separated into clusters') 
plt.show()

# 14 Let's run the code
# Let's take a look at the labeled datapoints, along with unassigned datapoints marked by solid points