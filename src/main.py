import argparse
import numpy as np

from data_loader import DataLoader
from distance import DistanceCalculator
from neighborhood import NeighborhoodBuilder
from clustering import Clustering

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
    print(f'k: {args.k}\n')

    # loading input data and preprocessing data
    print(f'Step 1: Loading and Preprocessing Data...')
    loader = DataLoader(args.dataset)
    _ = loader.load_data(limit=args.limit)
    data, num_cols, nom_cols = loader.preprocess_data()
    print(f'Loading and Preprocessing Complete.\n')

    # calculating distance matrix
    print(f'Step 2: Computing Distance Matrix...')
    distance_calculator = DistanceCalculator(num_cols, nom_cols)
    distance_matrix = distance_calculator.compute_distance_matrix(data)
    print(f'    Minimum Distance: {np.unique(distance_matrix)[1]:.5f}')
    print(f'    Maximum Distance: {distance_matrix.max():.5f}')
    print(f'Distance Matrix Computed.\n')

    # building neighborhoods
    print(f'Step 3: Building Neighborhoods...')
    nb_builder = NeighborhoodBuilder(distance_matrix, args.k)
    ndf_values, types, knb_mask = nb_builder.build_neighborhood()
    dp_count, ep_count, sp_count = (types == 'DP').sum(), (types == 'EP').sum(), (types == 'SP').sum()
    print(f'    Dense Points: {dp_count}')
    print(f'    Even Points: {ep_count}')
    print(f'    Sparse Points: {sp_count}')
    print(f'Neighborhood built.\n')

    # execute clustering
    print(f'Clustering...')
    clustering = Clustering(knb_mask, types)
    labels, n_clusters, n_noise = clustering.run_clustering()
    print(f'    Clusters Found: {n_clusters}')
    print(f'    Noise Points: {n_noise}')
    print(f'Clustering Complete.\n')

if __name__ == '__main__':
    main()