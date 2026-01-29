import argparse
import numpy as np
import time

from data_loader import DataLoader
from distance import DistanceCalculator
from neighborhood import NeighborhoodBuilder
from clustering import Clustering
from evaluation import Evaluator

# function for parsing input arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to the input dataset in CSV format.')
    parser.add_argument('--k', type=int, required=True, help='The k parameter for k-nearest neighbors.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows for testing.')

    return parser.parse_args()

def main():
    args = parse_arguments()

    # general informations
    print(f'\nNeighborhood-Based Clustering Algorithm\n')
    print(f'Dataset: {args.dataset}')

    target_column = None
    if args.dataset == 'data/flags.csv':
        target_column = 'religion'
    elif args.dataset == 'data/bank_marketing.csv':
        target_column = 'subscribe'
    elif args.dataset == 'data/wine_quality.csv':
        target_column = 'quality'
    else:
        print(f'Error: Unknown dataset \'{args.dataset}\'. No target column defined.')
    
    print(f'Target column: {target_column}')
    print(f'k: {args.k}\n')

    # loading input data and preprocessing data
    print(f'Step 1: Loading and Preprocessing Data...')
    loader = DataLoader(args.dataset)
    loader.load_data(target_column, limit=args.limit)
    data, num_cols, nom_cols = loader.preprocess_data()
    true_labels = loader.get_true_labels()
    print(f'Loading and Preprocessing Complete.\n')

    # starting performance time measuring
    print(f'Starting Performance Timer...\n')
    start_time = time.time()

    # calculating distance matrix
    print(f'Step 2: Computing Distance Matrix...')
    distance_calculator = DistanceCalculator(num_cols, nom_cols)
    distance_matrix = distance_calculator.compute_distance_matrix(data)
    print(f'    Minimum Distance: {np.unique(distance_matrix)[1]:.5f}')
    print(f'    Maximum Distance: {distance_matrix.max():.5f}')
    print(f'Distance Matrix Computed.\n')

    # building neighborhoods
    print(f'Step 3: Building Neighborhoods...')
    neighborhood_builder = NeighborhoodBuilder(distance_matrix, args.k)
    types, knb_mask = neighborhood_builder.build_neighborhood()
    dp_count, ep_count, sp_count = (types == 'DP').sum(), (types == 'EP').sum(), (types == 'SP').sum()
    print(f'    Dense Points: {dp_count}')
    print(f'    Even Points: {ep_count}')
    print(f'    Sparse Points: {sp_count}')
    print(f'Neighborhood built.\n')

    # execute clustering
    print(f'Clustering...')
    clustering = Clustering(knb_mask, types)
    n_clusters, n_noise = clustering.run_clustering()
    predicted_labels = clustering.get_predicted_labels()
    print(f'    Clusters Found: {n_clusters}')
    print(f'    Noise Points: {n_noise}')
    print(f'Clustering Complete.\n')

    # ending performance time measuring
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Algorithm Execution Time: {execution_time:.5f} seconds\n')

    # performance evaluation
    print(f'Evaluating Clustering...')
    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(distance_matrix, true_labels, predicted_labels)
    if metrics.get('silhouette') is not None:
        print(f'    Silhouette Coefficient: {metrics['silhouette']:.5f}')
        print(f'    (> 0.5 is good, close to 0 is overlapping, < 0 is incorrect)')
    else:
        print(f'    Silhouette Coefficient couldn\'t be computed - need at least 2 clusters to compute.')
    if metrics.get('ari') is not None:
        print(f'    Adjusted Rand Index: {metrics['ari']:.5f}')
        print(f'    (1 is perfect match with ground truth, 0 is random)')
    else:
        print(f'    Adjusted Rand Index couldn\'t be computed - need ground truth labels')
    print(f'    Generating Plots...')
    evaluator.plot_clusters(data, predicted_labels, num_cols)
    evaluator.plot_similarity_heatmap(distance_matrix, predicted_labels)
    print(f'Evaluation Clustering Complete.\n')

if __name__ == '__main__':
    main()