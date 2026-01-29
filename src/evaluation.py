import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

class Evaluator:
    def __init__(self):
        pass

    # computing evaluation metrics
    def compute_metrics(self, distance_matrix, true_labels, predicted_labels):
        results = {}

        mask = (predicted_labels != -1)
        labels_clean = predicted_labels[mask]
        distance_matrix_clean = distance_matrix[mask][:, mask]

        unique_labels = np.unique(labels_clean)
        n_clusters = len(unique_labels)

        # calculating Silhouette coefficient
        if n_clusters > 1:
            score = silhouette_score(distance_matrix_clean, labels_clean, metric='precomputed')
            results['silhouette'] = score
        else:
            results['silhouette'] = None

        # calculating Adjusted Rand index
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, predicted_labels)
            results['ari'] = ari

        return results
    
    # plotting clustering 
    def plot_clusters(self, data, labels, numerical_cols):
        num_data = data[numerical_cols].values
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(num_data)

        plt.figure()
        unique_labels = np.unique(labels)
        colours = sns.color_palette('tab10', len(unique_labels))

        # plotting noise
        noise_mask = (labels == -1)
        if np.sum(noise_mask) > 0:
            plt.scatter(reduced_data[noise_mask, 0], reduced_data[noise_mask, 1], c='lightgray', label='Noise', alpha=0.5, s=15)

        # plotting clusters
        for i, label in enumerate(unique_labels):
            if label == -1: 
                continue
            mask = (labels == label)
            plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], color=colours[i % len(colours)], label=f'Cluster #{label}', s=25)

        plt.title('NBC Clustering Results - PCA Projection')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    # plotting similarity heatmap
    def plot_similarity_heatmap(self, distance_matrix, labels):
        sorted_indices = np.argsort(labels)
        sorted_matrix = distance_matrix[sorted_indices][:, sorted_indices]

        plt.figure()
        sns.heatmap(sorted_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
        plt.title('Distance Matrix')
        plt.show()