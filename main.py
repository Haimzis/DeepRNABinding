import argparse
import os
import numpy as np
np.random.seed(24)
from models.selected_ngram_projection import SelectedNGramProjectionModel


def parse_args():
    parser = argparse.ArgumentParser(description='Compute correlations for RNA binding proteins using clustering.')

    # Positional arguments
    parser.add_argument('sequences_file', type=str, help='File containing the RNA sequences.')
    parser.add_argument('htr_selex_files', nargs='+', type=str, help='List of HTR-SELEX files (between 1 and 4 files).')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    rbp_model = SelectedNGramProjectionModel('data/htr-selex', os.path.join('data', args.sequences_file), None, KMer_LEN=(6, 8), top_k=1055, binary_embedding=True)
    predicted_scores = rbp_model.run(htr_selex_files=args.htr_selex_files)
    np.savetxt("bindings_intensities.txt", predicted_scores)
