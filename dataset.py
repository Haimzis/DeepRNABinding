import os
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a mapping from nucleotide to one-hot encoding
nucleotide_to_onehot = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0.25, 0.25, 0.25, 0.25]
}

def encode_sequence(sequence):
    """Encodes a nucleotide sequence into a one-hot encoded format.

    Args:
        sequence (str): A string representing the nucleotide sequence.

    Returns:
        list: A list of one-hot encoded vectors for the nucleotide sequence.
    """
    try:
        return [nucleotide_to_onehot[nuc] for nuc in sequence]
    except KeyError:
        raise KeyError(f"Unknown nucleotide encountered in sequence: {sequence}")

class RNASequenceDataset(Dataset):
    def __init__(self, sequences_file, intensities_dir, htr_selex_dir, i=1, train=True, trim=False, negative_examples=0):
        """Initializes the RNASequenceDataset.

        Args:
            sequences_file (str): Path to the file containing RNA sequences.
            intensities_dir (str): Directory containing intensity files.
            htr_selex_dir (str): Directory containing htr-selex files.
            i (int): Index of the RBP file.
            train (bool): Whether the dataset is for training or testing.
            trim (bool): Whether to trim the data for faster debugging.
            negative_examples (int): The number of negative samples to generate.
        """
        self.sequences = self.load_sequences(sequences_file)
        self.intensities_dir = intensities_dir
        self.htr_selex_dir = htr_selex_dir
        self.i = i
        self.train = train
        self.intensities = self.load_intensities()
        self.data = self.load_data(trim, negative_examples)
        self.process_data()

    def load_sequences(self, file_path):
        """Loads all possible RNA sequences from a file.

        Args:
            file_path (str): Path to the file containing RNA sequences.

        Returns:
            list: A list of RNA sequences.
        """
        with open(file_path, 'r') as f:
            sequences = f.read().splitlines()
        return sequences

    def load_intensities(self):
        """Loads intensities for the specified RBP index.

        Returns:
            np.ndarray: An array of intensity values or None if not in training mode or file doesn't exist.
        """
        intensities = None
        intensities_file = os.path.join(self.intensities_dir, f'RBP{self.i}.txt')
        if os.path.exists(intensities_file):
            intensities = pd.read_csv(intensities_file, header=None).values.flatten()
        return intensities

    def load_data(self, trim, negative_examples):
        """Loads data from htr-selex files for the specified RBP index.

        Returns:
            np.recarray: A record array containing sequences, occurrences, and labels.
        """
        htr_selex_files = [os.path.join(self.htr_selex_dir, f'RBP{self.i}_{j}.txt') for j in range(1, 5)]
        df_list = []
        nrows=100 if trim else None

        for j, file in enumerate(htr_selex_files, start=1):
            if os.path.exists(file):
                df = pd.read_csv(file, header=None, names=['sequence', 'occurrences'], nrows=nrows)
                df['occurrences'] = df['occurrences'].astype(np.uint8)
                df['label'] = j
                df_list.append(df)

        if not df_list:
            raise FileNotFoundError(f"No htr-selex files found for RBP{self.i}")

        combined_df = pd.concat(df_list, ignore_index=True)

        if negative_examples > 0:
            neg_examples = [{'sequence': ''.join(np.random.choice(['A', 'C', 'G', 'T'], 40)),
                             'occurrences': 1, 'label': 0} for _ in range(negative_examples)]
            neg_df = pd.DataFrame(neg_examples)
            combined_df = pd.concat([combined_df, neg_df], ignore_index=True)

        if trim:
            combined_df = combined_df.head(100)

        combined_df['sequence'] = combined_df['sequence'].astype(str)
        combined_df['label'] = combined_df['label'].astype(np.uint8)

        data = combined_df.to_records(index=False)
        return data

    def process_data(self):
        """Processes the loaded data to get maximum labels for each sequence."""
        df = pd.DataFrame(self.data)
        max_labels = df.groupby('sequence').agg({
            'occurrences': 'sum',
            'label': 'max'
        }).reset_index()
        self.data = max_labels.to_records(index=False)

    def create_test_loader(self, batch_size=32):
        """Creates a test data loader for the dataset.

        Args:
            batch_size (int, optional): The batch size for the data loader. Defaults to 32.

        Returns:
            DataLoader: The test data loader.
        """
        df = pd.DataFrame({'sequence': self.sequences})
        df['encoded'] = df['sequence'].apply(lambda seq: encode_sequence(seq))

        max_length = 41
        df['padded'] = df['encoded'].apply(
            lambda seq: seq + [[0.25, 0.25, 0.25, 0.25]] * (max_length - len(seq)) if len(seq) < max_length else seq
        )

        encoded_sequences = np.array(df['padded'].tolist())
        encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.float32)
        dataset = TensorDataset(encoded_sequences)

        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_person_correlation(self, predictions):
        """Calculates the Pearson correlation between the predictions and intensities.

        Args:
            predictions (np.ndarray): The predictions from the model.

        Returns:
            float: The Pearson correlation between the predictions and intensities.
        """
        return np.corrcoef(predictions, self.intensities)[0, 1]

    def is_same_length(self):
        """Checks if all sequences in self.data have the same length."""
        return len(set(len(record['sequence']) for record in self.data)) == 1

    def get_sequence_length(self):
        """Returns the length of the RNA sequences.

        Returns:
            int: The length of the RNA sequences.
        """
        return len(self.data[0]['sequence'])

    def get_sequences(self):
        """Returns the list of RNA sequences.

        Returns:
            list: The list of RNA sequences.
        """
        return self.sequences

    def get_intensities(self):
        """Returns the array of intensity values.

        Returns:
            np.ndarray: The array of intensity values or None.
        """
        return self.intensities

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The number of records in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Gets a data sample by index.

        Args:
            idx (int): The index of the data sample.

        Returns:
            tuple: A tuple containing the one-hot encoded sequence, occurrences, and label as tensors.
        """
        record = self.data[idx]
        sequence_encoded = torch.tensor(encode_sequence(record['sequence']), dtype=torch.float32, device=device)
        occurrences = torch.tensor(record['occurrences'], dtype=torch.int64, device=device)
        label = torch.tensor(record['label'], dtype=torch.int64, device=device)
        return sequence_encoded, occurrences, label

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

def compute_baseline(sequences_file, intensities_dir, htr_selex_dir):
    # Calculate correlations
    correlations = []
    for i in range(38):
        last_cycle_path = os.path.join(htr_selex_dir, f'RBP{i + 1}_4.txt') if os.path.exists(os.path.join(htr_selex_dir, f'RBP{i + 1}_4.txt')) else os.path.join(htr_selex_dir, f'RBP{i + 1}_3.txt')
        rnacompete_path = sequences_file
        binding_intensities_path = os.path.join(intensities_dir, f'RBP{i + 1}.txt')
        correlations.append(compute_baseline_helper(last_cycle_path, rnacompete_path, binding_intensities_path))
        print(f"Correlation for RBP{i + 1}: {correlations[i]}")


    # Save correlations to a file
    with open('baseline_correlations.txt', 'w') as file:
        for correlation in correlations:
            file.write(f'{correlation}\n')

    print(f"Correlations saved to 'baseline_correlations.txt'")

def compute_baseline_helper(last_cycle_path, rnacompete_path, binding_intensities_path):
    # Generate all possible 7-mers, 4^7 = 16384
    all_7mers = [''.join(combo) for combo in product('ACGT', repeat=7)]

    # Count 7-mers in the last cycle file
    vectorizer = CountVectorizer(vocabulary=all_7mers, analyzer='char', ngram_range=(7, 7), lowercase=False)

    with open(last_cycle_path, 'r') as file:
        sequences = [line.strip().split(',')[0] for line in file]

    """
    # Read sequences and their occurrences, then repeat sequences based on occurrences
    sequences = []
    with open(last_cycle_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                seq, count = parts[0], parts[1]
                try:
                    count = int(count)
                    sequences.extend([seq] * count)
                except ValueError:
                    print(f"Warning: Invalid count '{count}' for sequence '{seq}'. Skipping.")
            else:
                print(f"Warning: Invalid line format: {line.strip()}. Skipping.")
    """

    counts = vectorizer.fit_transform(sequences)
    seven_mer_scores = np.array(counts.sum(axis=0)).flatten()

    # Score RNAcompete sequences
    with open(rnacompete_path, 'r') as file:
        rnacompete_sequences = [line.strip().split()[0] for line in file]

    # Use CountVectorizer to efficiently extract and count 7-mers
    rnacompete_counts = vectorizer.fit_transform(rnacompete_sequences)

    # Compute scores for each sequence
    rnacompete_scores = rnacompete_counts.dot(seven_mer_scores)
    seq_lens = np.array([len(seq) - 6 for seq in rnacompete_sequences])
    rnacompete_scores = rnacompete_scores / seq_lens

    # Load binding intensities
    binding_intensities = np.loadtxt(binding_intensities_path)

    # Compute Pearson correlation
    correlation, _ = pearsonr(rnacompete_scores, binding_intensities)

    return correlation


if __name__ == '__main__':
    # Load training and testing datasets for a specific RBP{i}
    sequences_file = 'data/RNAcompete_sequences_rc.txt'
    intensities_dir = 'data/RNAcompete_intensities'
    htr_selex_dir = 'data/htr-selex'

    i = 1  # Set i to 1 for now

    train_dataset = RNASequenceDataset(sequences_file, intensities_dir, htr_selex_dir, i, train=True)

    test_dataset = RNASequenceDataset(sequences_file, intensities_dir, htr_selex_dir, i, train=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Example usage of new methods
    for batch in train_loader:
        sequences, occurrences, labels = batch
        print(sequences.shape, sequences.device)  # Should print (32, Sequence Length, Embedding Length)