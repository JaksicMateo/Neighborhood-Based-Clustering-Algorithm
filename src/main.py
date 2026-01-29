import argparse
import os
import sys

from data_loader import DataLoader
from distance import DistanceCalculator

# function for parsing input arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to the input dataset in CSV format.')
    parser.add_argument('--k', type=int, required=True, help='The k parameter for k-nearest neighbors.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows for testing.')

    return parser.parse_args()

def main():
    args = parse_arguments()

    print(f'\nNeighborhood-Based Clustering Algorithm\n')
    print(f'Dataset: {args.dataset}')
    print(f'k: {args.k}\n')

    print(f'Step 1: Loading and Preprocessing Data...')
    loader = DataLoader(args.dataset)
    _ = loader.load_data(limit=args.limit)
    data, num_cols, nom_cols = loader.preprocess_data()
    print(f'Loading and Preprocessing Complete.\n')

    print(f'Step 2: Computing Distance Matrix...')
    distance_calculator = DistanceCalculator(num_cols, nom_cols)
    distance_matrix = distance_calculator.compute_distance_matrix(data)
    print(f'Distance Matrix Computed\n')
    print(f'{distance_matrix[0,5]:.4f}')




if __name__ == '__main__':
    main()