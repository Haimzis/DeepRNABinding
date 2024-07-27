import argparse

def parse_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='DeepSELEX model for predicting intensity levels.')
    parser.add_argument('--rbp_num', type=int, default=1, help='The number of the RNA binding protein to predict.')
    parser.add_argument('--predict', type=bool, default=False ,help='Predict the intensity levels.')
    parser.add_argument('--sequences_file', type=str, default='data/RNAcompete_sequences_rc.txt', help='File containing the RNA sequences.')
    parser.add_argument('--intensities_dir', type=str, default='data/RNAcompete_intensities', help='Directory containing the intensity levels.')
    parser.add_argument('--htr_selex_dir', type=str, default='data/htr-selex', help='Directory containing the HTR-SELEX documents.')
    parser.add_argument('--predict_output_dir', type=str, default='outputs/predictions/Deep_SELEX', help='Directory to save the predictions.')
    parser.add_argument('--save_model_file', type=str, default='outputs/models/Deep_SELEX', help='Directory to save the model.')
    parser.add_argument('--load_model_file', type=str, default='outputs/models/Deep_SELEX/best_model.ckpt', help='File to load the model.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training.')
    parser.add_argument('--early_stopping', type=int, default=10, help='Number of epochs for early stopping.')
    parser.add_argument('--seed', type=int, default=24, help='Seed for random number generator.')
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds for k-fold cross-validation.')
    parser.add_argument('--trim', type=bool, default=False, help='Whether to trim the data for faster debugging.')
    parser.add_argument('--negative_examples', type=int, default=0, help='Number of negative samples to generate.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save the logs.')

    return parser.parse_args()
