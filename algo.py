#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import datetime
import log


# #  KMeans Algorithm Implementation

# In[12]:


np.random.seed(42)
logger =  log.Logger().get_logger("INFO", "clustering_main_log")
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KMeans():
    logger.info("Started at Kmeans algorithm at %s" % str(datetime.datetime.now()))
    def __init__(self, K, max_iters, plot_steps=True):
      try: 
        self.K = K
        logger.info("K value for Kmeans: " + str(self.K) )
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        # list of sample indices for each cluster 
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []
      except Exception as e:
        logger.error(str(e))
    def predict(self, X):
      try:
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break
            if self.plot_steps:
                self.plot()
        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)
      except Exception as e:
        logger.error(str(e))
    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it is assigned to 
      try:
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels
      except Exception as e:
        logger.error(str(e))
    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
      try:
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
      except Exception as e:
        logger.error(str(e))
    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
      try:
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index
      except Exception as e:
        logger.error(str(e))
    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
      try:
        # Assigning numpy arrays of zeroes and assigning it to the centers
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
      except Exception as e:
        logger.error(str(e))
    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, for all centroids
      try:
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
      except Exception as e:
        logger.error(str(e))
    def plot(self):
      try:
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)
        plt.show()
      except Exception as e:
        logger.error(str(e))
    def cent(self):
        return self.centroids
    


# # DBScan Algorithm Implementation

# In[14]:



def MaximumDistance(P,Q):
  try:
    intermediateValues = []
    for i in range(len(P[2])):
        intermediateValues.append(abs(Q[2][i]-P[2][i]))
    return max(intermediateValues)
  except Exception as e:
    logger.error(str(e))


# Finding neighbor points for a chosen point
def FindNeighbours(Point, Points, eps):
  try:
    tempNeighbours = []
    for y in range(len(Points)):
        for x in range(len(Points[0])):
            if MaximumDistance(Point, Points[y][x]) <= eps:
                    tempNeighbours.append(Points[y][x])
    return tempNeighbours
  except Exception as e:
    logger.error(str(e))

# reads vector array, performs dbscan and outputs vector array
def dbscan(vectors: list, minpts: int, epsilon: int) -> list:
    #Initialization
  logger.info("Started at DBscan algorithm at %s" % str(datetime.datetime.now()))
  logger.info("minpts: " + str(minpts))
  logger.info("epsilon:" + str(epsilon))
  try:
    pointsArray = []
    for y in range(len(vectors)):
        pointsArray.append([])
        for x in range(len(vectors[0])):
            pointsArray[y].append([y,x,vectors[y][x],"Undefined"])
            
    # DBSCAN clustering
    clusterCounter = 0
    for y in range(len(vectors)):
        for x in range(len(vectors[0])):
            if pointsArray[y][x][-1] != "Undefined":
                continue
            Neighbours = FindNeighbours(pointsArray[y][x], pointsArray, epsilon)
            if len(Neighbours) < minpts:
                pointsArray[y][x][-1] = "Noise"
                continue

            clusterCounter = clusterCounter + 1
            pointsArray[y][x][-1] = str(clusterCounter)
            if pointsArray[y][x] in Neighbours:
                Neighbours.remove(pointsArray[y][x])

            for innerPoint in Neighbours:
                if innerPoint[-1] == "Noise":
                    pointsArray[innerPoint[0]][innerPoint[1]][-1] = str(clusterCounter)
                if innerPoint[-1] != "Undefined":
                    continue
                pointsArray[innerPoint[0]][innerPoint[1]][-1] = str(clusterCounter)
                NeighboursInner = FindNeighbours(innerPoint, pointsArray, epsilon)
                if len(NeighboursInner) >= minpts:
                    Neighbours.append(NeighboursInner)              
    # Getting distinct clusters

    clusterNumbers = []
    for y in range(len(vectors)):
        for x in range(len(vectors[0])):
            if pointsArray[y][x][-1] not in clusterNumbers:
                clusterNumbers.append(pointsArray[y][x][-1])

    # Mapping cluster's averages
    averagesForClusters = []
    for item in clusterNumbers:
        n = 0
        vectorTemps = [0]*len(pointsArray[0][0][2])
        for y in range(len(vectors)):
            for x in range(len(vectors[0])):
                if pointsArray[y][x][-1] == item:
                    for i in range(len(pointsArray[y][x][2])):
                        vectorTemps[i] = vectorTemps[i] + pointsArray[y][x][2][i]
                    n = n + 1
        # Checking if vector us not Zero 
        for i in range(len(vectorTemps)):
            if vectorTemps[i] != 0:
                vectorTemps[i] = vectorTemps[i]/n
        averagesForClusters.append(vectorTemps)

    # Creating array of clusters and changing cluster averages with initial values

    clusteredVectors = []
    for y in range(len(pointsArray)):
        clusteredVectors.append([])
        for x in range(len(pointsArray[0])):
            clusteredVectors[y].append(averagesForClusters[clusterNumbers.index(pointsArray[y][x][-1])])
    return clusteredVectors
  except Exception as e:
    logger.error(str(e))





