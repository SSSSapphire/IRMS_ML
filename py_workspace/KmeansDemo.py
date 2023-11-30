from sklearn.cluster import KMeans
import numpy as np

kmeans = KMeans(n_clusters = 2, random_state = 0,).fit(X)