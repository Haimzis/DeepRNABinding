import args
import logging as log
from models.baseline import BaselineModel

def main(args):
    # Set up logging
    log.basicConfig(level=log.INFO)

    # Initialize Baseline Model
    baseline_model = BaselineModel(args.htr_selex_dir, args.sequences_file, args.intensities_dir)

    # Compute baseline correlations
    correlations = baseline_model.compute_baseline(num_rbps=args.rbp_num)
    log.info(f"Computed baseline correlations: {correlations}")

if __name__ == '__main__':
    args = args.parse_args()
    main(args)
