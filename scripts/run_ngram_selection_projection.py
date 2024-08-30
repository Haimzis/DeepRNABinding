import argparse
import logging as log

import numpy as np
from models.selected_ngram_projection import SelectedNGramProjectionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Compute correlations for RNA binding proteins using clustering.')
    parser.add_argument('--rbp_num', type=int, default=38, help='The number of the RNA binding proteins to process.')
    parser.add_argument('--sequences_file', type=str, default='data/RNAcompete_sequences_rc.txt', help='File containing the RNA sequences.')
    parser.add_argument('--intensities_dir', type=str, default='data/RNAcompete_intensities', help='Directory containing the intensity levels.')
    parser.add_argument('--htr_selex_files', type=str, required=True, help='Directory containing the HTR-SELEX documents or a space-separated list of specific files.')
    return parser.parse_args()


def main(args):
    # Set up logging
    log.basicConfig(level=log.INFO)

    # Initialize RBP Model
    rbp_model = SelectedNGramProjectionModel(args.htr_selex_files, args.sequences_file, args.intensities_dir, KMer_LEN=(6, 8), top_k=2048, binary_embedding=True, n_jobs=1)

    # Compute correlations
    correlations = rbp_model.run_all(num_rbps=args.rbp_num)
    log.info(f"Computed correlations: {correlations}")
    log.info(f"avg correlation: {np.array(correlations).mean()}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
