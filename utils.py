from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import logging as log
import seaborn as sns
from datasets.rna_sequence_dataset import RNASequenceDataset

# TODO: dont put here stuff for training, just utillities. 

def save_predictions(predictions, predict_output_dir):
    """
    Save the predictions to a file.
    Args:
        predictions: A tensor containing the predictions.
        predict_output_dir: The directory to save the predictions to.
    """
    if not os.path.exists(predict_output_dir):
        os.makedirs(predict_output_dir)
    with open(os.path.join(predict_output_dir, 'predictions.txt'), 'w') as f:
        for pred in predictions:
            f.write(f'{pred}\n')

def get_device():
    """
    Get the device (GPU or CPU) for computation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    return device

def generate_rbp_intensity_correlation_heatmap(sequences_file, intensities_dir, htr_selex_dir, num_rbps=38, output_file='rbp_intensity_correlations_heatmap.png'):
    """
    Generate a heatmap of correlations between RBP intensities.

    Args:
        sequences_file (str): Path to the file containing RNA sequences.
        intensities_dir (str): Directory containing intensity files.
        htr_selex_dir (str): Directory containing htr-selex files.
        num_rbps (int): Number of RBPs to include in the analysis.
        output_file (str): Name of the output file for the heatmap.

    Returns:
        None
    """
    # Create datasets for all RBPs
    datasets = [RNASequenceDataset(sequences_file, intensities_dir, htr_selex_dir, i, train=True) for i in range(1, num_rbps + 1)]

    # Calculate correlations
    correlations = np.zeros((num_rbps, num_rbps))
    for i in range(num_rbps):
        for j in range(num_rbps):
            intensities_i = datasets[i].get_intensities()
            intensities_j = datasets[j].get_intensities()
            if intensities_i is not None and intensities_j is not None:
                correlations[i][j] = np.corrcoef(intensities_i, intensities_j)[0, 1]
            else:
                correlations[i][j] = np.nan

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of RBP Intensities')
    plt.xlabel('RBP Index')
    plt.ylabel('RBP Index')
    plt.savefig(output_file)
    plt.close()

    print(f"Heatmap saved as '{output_file}'")

def get_person_correlation(self, predictions, intensities):
    """Calculates the Pearson correlation between the predictions and intensities."""
    return np.corrcoef(predictions, intensities)[0, 1]