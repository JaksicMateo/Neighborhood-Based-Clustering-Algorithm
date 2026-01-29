import numpy as np

class NeighborhoodBuilder:
    def __init__(self, distance_matrix, k):
        self.distance_matrix = distance_matrix
        self.k = k
        self.n_samples = distance_matrix.shape[0]

    # function for building neighborhoods
    def build_neighborhood(self):
        sorted_indices = np.argsort(self.distance_matrix, axis = 1)
        # instead of getting k closest objects, we get distance of the kth closest object
        kth_distance = np.take_along_axis(self.distance_matrix, sorted_indices[:, self.k][:, None], axis=1)
        
        # in case there are more then one kth closest object, we take them all
        knb_mask = self.distance_matrix <= kth_distance
        np.fill_diagonal(knb_mask, False)

        knb_counts = knb_mask.sum(axis=1)
        rknb_counts = knb_mask.sum(axis=0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ndf_values = rknb_counts / knb_counts
            ndf_values = np.nan_to_num(ndf_values, nan=0.0)

        # classify points based on ndf value
        point_types = np.array(['SP'] * self.n_samples, dtype=object)
        point_types[ndf_values >= 1.000001] = 'DP'
        point_types[np.isclose(ndf_values, 1.0)] = 'EP'

        return ndf_values, point_types, knb_mask