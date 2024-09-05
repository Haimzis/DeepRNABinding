from turtle import pd
from matplotlib import pyplot as plt
import numpy as np
from datasets.rna_sequence_dataset import RNASequenceDataset
import seaborn as sns
import os


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

def get_min_max_intensities(intensities_dir):
    intensities = []
    max_arr = []
    min_arr = []
    for i in range(1, 39):
        intensities_file = os.path.join(intensities_dir, f'RBP{i}.txt')
        if os.path.exists(intensities_file):
            intensities.append(pd.read_csv(intensities_file, header=None).values.flatten())
            max_arr.append(np.max(intensities[-1]))
            min_arr.append(np.min(intensities[-1]))
    with open('min_max_intensities.txt', 'w') as f:
        for i in range(38):
            f.write(f'RBP{i+1}: Min={min_arr[i]}, Max={max_arr[i]}\n')
        f.write(f'Min: {min(min_arr)}\n')
        f.write(f'Max: {max(max_arr)}\n')


def get_intensity_bins(intensities_dir):
    bins = []
    ranges = []
    for i in range(1, 39):
        intensities_file = os.path.join(intensities_dir, f'RBP{i}.txt')
        if os.path.exists(intensities_file):
            intensities = pd.read_csv(intensities_file, header=None).values.flatten()
            # Divide intensities into 5 classes
            hist, bin_edges = np.histogram(intensities, bins=5)
            # Calculate ranges and count items in each class
            class_info = []
            for i in range(5):
                class_range = (bin_edges[i], bin_edges[i + 1])
                count = hist[i]
                class_info.append((class_range, count))
            bins.append(hist)
            ranges.append(class_info)
    with open('intensity_bins.txt', 'w') as f:
        for i in range(38):
            f.write(f'RBP{i+1}: {bins[i]}\n')
            # Calculate the percentage for each element in bins[i]
            percentages = (bins[i] / np.sum(bins[i])) * 100
            # Create a string that contains all percentages formatted to 2 decimal places
            percentages_str = ', '.join([f'{p:.2f}%' for p in percentages])
            # Write the formatted string to the file
            f.write(f'RBP{i + 1}: {percentages_str}\n')
            f.write(f'Ranges: {ranges[i]}\n')