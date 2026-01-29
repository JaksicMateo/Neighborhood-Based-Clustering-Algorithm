import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class DistanceCalculator:
    def __init__(self, numerical_cols, nominal_cols):
        self.numerical_cols = numerical_cols
        self.nominal_cols = nominal_cols

    # calculating overall distance
    def compute_distance_matrix(self, data):
        n_samples = len(data)
        total_dist_sq = np.zeros((n_samples, n_samples))

        # calculating numerical distances
        if self.numerical_cols:
            num_data = data[self.numerical_cols].values.astype(float)
            total_dist_sq += euclidean_distances(num_data, squared=True)

        # calculating nominal distances
        if self.nominal_cols:
            nom_data = data[self.nominal_cols].values
            for i in range(nom_data.shape[1]):
                col_values = nom_data[:, i].reshape(-1, 1)
                mismatches = (col_values != col_values.T).astype(float)
                total_dist_sq += mismatches

        return np.sqrt(total_dist_sq)