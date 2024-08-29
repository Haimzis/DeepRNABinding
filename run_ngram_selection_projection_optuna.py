import argparse
import optuna
import os
from models.selected_ngram_projection import SelectedNGramProjectionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Compute correlations for RNA binding proteins using clustering.')
    parser.add_argument('--rbp_num', type=int, default=38, help='The number of the RNA binding proteins to process.')
    parser.add_argument('--sequences_file', type=str, default='data/RNAcompete_sequences_rc.txt', help='File containing the RNA sequences.')
    parser.add_argument('--intensities_dir', type=str, default='data/RNAcompete_intensities', help='Directory containing the intensity levels.')
    parser.add_argument('--htr_selex_dir', type=str, default='data/htr-selex', help='Directory containing the HTR-SELEX documents.')

    return parser.parse_args()

# Define the objective function for optimization
def objective(trial):
    # Suggest hyperparameters
    kmer_len_start = trial.suggest_int('kmer_len_start', 4, 7)
    kmer_len_end = trial.suggest_int('kmer_len_end', 7, 8)
    top_k = trial.suggest_int('top_k', 1024, 16_384)
    binary_embedding = trial.suggest_categorical('binary_embedding', [True, False])

    # Initialize the SelectedNGramProjectionModel with the suggested hyperparameters
    rbp_model = SelectedNGramProjectionModel(
        htr_selex_dir=args.htr_selex_dir,
        sequences_file=args.sequences_file,
        intensities_dir=args.intensities_dir,
        KMer_LEN=(kmer_len_start, kmer_len_end),
        top_k=top_k,
        binary_embedding=binary_embedding
    )

    # Compute correlations (average correlation across all RBPs)
    correlations = rbp_model.run_all(num_rbps=args.rbp_num)
    avg_correlation = sum(correlations) / len(correlations)

    # Log the hyperparameters and results
    log_results(trial.number, kmer_len_start, kmer_len_end, top_k, binary_embedding, avg_correlation)

    return avg_correlation

# Function to log the results of each trial
def log_results(trial_number, kmer_len_start, kmer_len_end, top_k, binary_embedding, avg_correlation):
    # Create the results directory if it doesn't exist
    results_dir = 'optuna_results'
    os.makedirs(results_dir, exist_ok=True)

    # Construct the filename
    filename = os.path.join(results_dir, 'trial_results.txt')

    # Write the trial results to the file
    with open(filename, 'a') as f:
        f.write(f'Trial {trial_number}:\n')
        f.write(f'  kmer_len_start: {kmer_len_start}\n')
        f.write(f'  kmer_len_end: {kmer_len_end}\n')
        f.write(f'  top_k: {top_k}\n')
        f.write(f'  binary_embedding: {binary_embedding}\n')
        f.write(f'  avg_correlation: {avg_correlation}\n')
        f.write('-' * 40 + '\n')

if __name__ == '__main__':
    args = parse_args()

    # Create the Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50, n_jobs=1)

    # Print the best hyperparameters found
    print(f"Best hyperparameters: {study.best_params}")
