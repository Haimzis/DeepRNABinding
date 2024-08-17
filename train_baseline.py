import argparse
import logging as log
from models.baseline import BaselineModel


def parse_args():
    parser = argparse.ArgumentParser(description='Compute baseline correlations for RNA binding proteins.')
    parser.add_argument('--rbp_num', type=int, default=38, help='The number of the RNA binding protein to process.')
    parser.add_argument('--sequences_file', type=str, default='data/RNAcompete_sequences_rc.txt', help='File containing the RNA sequences.')
    parser.add_argument('--intensities_dir', type=str, default='data/RNAcompete_intensities', help='Directory containing the intensity levels.')
    parser.add_argument('--htr_selex_dir', type=str, default='data/htr-selex', help='Directory containing the HTR-SELEX documents.')

    return parser.parse_args()

def main(args):
    # Set up logging
    log.basicConfig(level=log.INFO)

    # Initialize Baseline Model
    baseline_model = BaselineModel(args.htr_selex_dir, args.sequences_file, args.intensities_dir)

    # Compute baseline correlations
    correlations = baseline_model.compute_baseline(num_rbps=args.rbp_num)
    log.info(f"Computed baseline correlations: {correlations}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
