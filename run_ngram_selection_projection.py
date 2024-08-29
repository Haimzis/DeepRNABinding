import argparse
import logging as log
import os
import re

import numpy as np
np.random.seed(24)
from models.selected_ngram_projection import SelectedNGramProjectionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Compute correlations for RNA binding proteins using clustering.')
    parser.add_argument('--rbp_num', type=int, default=38, help='The number of the RNA binding proteins to process.')
    parser.add_argument('--sequences_file', type=str, default='data/RNAcompete_sequences_rc.txt', help='File containing the RNA sequences.')
    parser.add_argument('--intensities_dir', type=str, default='data/RNAcompete_intensities', help='Directory containing the intensity levels.')
    parser.add_argument('--htr_selex_dir', type=str, default='data/htr-selex', help='Directory containing the HTR-SELEX documents.')
    parser.add_argument('--htr_selex_files', type=str, nargs='+', help='Sequence of RBP\'s files')
    return parser.parse_args()

def main(args):
    # Set up logging
    log.basicConfig(level=log.INFO)

    # Initialize RBP Model
    rbp_model = SelectedNGramProjectionModel(args.htr_selex_dir, args.sequences_file, args.intensities_dir, KMer_LEN=(6, 8), top_k=1055, binary_embedding=True, n_jobs=10)

    # # Compute all correlations
    # correlations = rbp_model.run_all(num_rbps=args.rbp_num)
    # log.info(f"Computed correlations: {correlations}")
    # log.info(f"avg correlation: {np.array(correlations).mean()}")
    
    rna_scores = rbp_model.run(htr_selex_files=args.htr_selex_files)
    rbp_number = re.search(r'RBP\{(\d+)\}', args.htr_selex_files[0]).group(1)
    correlation = rbp_model.compute_correlation(rna_scores, os.path.join(args.intensities_dir, f'RBP{rbp_number}.txt'))
    log.info(f"Computed correlation: {correlation}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
