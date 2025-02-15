#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 13:24:02 2023

@author: lbk
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

from scipy.cluster.hierarchy import dendrogram
# Create linkage matrix and then plot the dendrogram
def plot_dendrogram(model, **kwargs):
# Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 # leaf node
            else:
                current_count += counts[child_idx - n_samples]
            counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    
    
# Generate data
x, y = make_blobs(n_samples = 100, n_features = 2, centers = 5)
plt.rcParams['figure.dpi'] = 200
plt.scatter(x[:,0], x[:,1], c = 'black', alpha = 0.5)
plt.show()
# Run K-Means
mdl = KMeans(n_clusters = 5).fit(x)
centroids = mdl.cluster_centers_
pred = mdl.labels_
# Plot points by cluster and cluster centroids

plt.rcParams['figure.dpi'] = 200
for i in range(5):
    plt.scatter(x[pred == i, 0], x[pred == i, 1], alpha = 0.5)

plt.scatter(centroids[:,0], centroids[:,1], c = 'red')
plt.show()



import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
# Generate data
x, y = make_blobs(n_samples = 100, n_features = 2, centers = 5)
plt.rcParams['figure.dpi'] = 200
plt.scatter(x[:,0], x[:,1], c = 'black', alpha = 0.5)
plt.show()
# Run K-Means
#mdl = AgglomerativeClustering(n_clusters = 5, linkage = 'average').fit(x)
mdl = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage = 'average').fit(x)

pred = mdl.labels_
dist = mdl.distances_
# Plot points by cluster and cluster centroids
plt.rcParams['figure.dpi'] = 200
for i in range(5):
    plt.scatter(x[pred == i, 0], x[pred == i, 1], alpha = 0.5)
plt.show()

plot_dendrogram(mdl)


