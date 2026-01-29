import numpy as np
from collections import deque

class Clustering:
    def __init__(self, knb_mask, point_types):
        self.knb_mask = knb_mask
        self.point_types = point_types
        self.n_samples = len(point_types)
        # we label all samples as noise (-1)
        self.labels = np.full(self.n_samples, -1, dtype=int)
        self.cluster_id = 0

    # function for executing clustering process
    def run_clustering(self):
        # iterate through every point in the dataset and start clusters
        for i in range(self.n_samples):
            if self.labels[i] == -1 and self.point_types[i] in ['DP', 'EP']:
                self.expand_cluster(i)
                self.cluster_id += 1

        # remaining samples are noise
        n_clusters = self.cluster_id
        n_noise = np.sum(self.labels == -1)

        return self.labels, n_clusters, n_noise
    
    # function for expanding cluster
    def expand_cluster(self, seed_index):
        # assigning cluster ID
        self.labels[seed_index] = self.cluster_id

        # queue for points that need to be visited
        queue = deque([seed_index])
        while queue:
            current_point = queue.popleft()

            # find all neighbors in current point
            neighbors = np.where(self.knb_mask[current_point])[0]

            for neighbor in neighbors:
                # if neighbor is already part of a cluster
                if self.labels[neighbor] != -1:
                    continue

                # assign neighbor to the current cluster
                self.labels[neighbor] = self.cluster_id

                # if neighbor if a core point (DP, EP) we add it to queue to continue expanding from there
                # if it's sparse point then we don't add it to the queue and stop expansion path
                
                if self.point_types[neighbor] in ['DP', 'EP']:
                    queue.append(neighbor)