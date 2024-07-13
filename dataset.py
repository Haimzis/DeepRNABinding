import copy
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold

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
    except:
        raise KeyError(f"Unknown nucleotide encountered in sequence: {sequence}")

class RNASequenceDataset(Dataset):
    def __init__(self, sequences_file, intensities_dir, htr_selex_dir, i=1, train=True):
        """Initializes the RNASequenceDataset.

        Args:
            sequences_file (str): Path to the file containing RNA sequences.
            intensities_dir (str): Directory containing intensity files.
            htr_selex_dir (str): Directory containing htr-selex files.
            i (int): Index of the RBP file.
            train (bool): Whether the dataset is for training or testing.
        """
        self.sequences = self.load_sequences(sequences_file)
        self.intensities_dir = intensities_dir
        self.htr_selex_dir = htr_selex_dir
        self.i = i
        self.train = train
        self.intensities = self.load_intensities()
        self.data = self.load_data()
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
        if self.train:
            intensities_file = os.path.join(self.intensities_dir, f'RBP{self.i}.txt')
            if os.path.exists(intensities_file):
                intensities = pd.read_csv(intensities_file, header=None).values.flatten()
        return intensities
    
    def load_data(self):
        """Loads data from htr-selex files for the specified RBP index.

        Returns:
            np.recarray: A record array containing sequences, occurrences, and labels.
        """
        htr_selex_files = [os.path.join(self.htr_selex_dir, f'RBP{self.i}_{j}.txt') for j in range(1, 5)]
        df_list = []

        for j, file in enumerate(htr_selex_files, start=1):
            if os.path.exists(file):
                df = pd.read_csv(file, header=None, names=['sequence', 'occurrences'])
                df['occurrences'] = df['occurrences'].astype(np.uint8)
                df['label'] = j
                df_list.append(df)

        if not df_list:
            raise FileNotFoundError(f"No htr-selex files found for RBP{self.i}")

        if len(df_list) == 1:
            data = df_list[0].to_records(index=False)
        else:
            combined_df = pd.concat(df_list, ignore_index=True)
            
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

    def create_k_fold_loaders(self, kfold, iteration, batch_size=32, trim=False):
        """Creates train and validation data loaders for a specific k-fold iteration.

        Args:
            kfold (int): The number of folds for k-fold cross-validation.
            iteration (int): The current fold iteration (0-based).
            batch_size (int, optional): The batch size for the data loaders. Defaults to 32.
            trim (bool, optional): Whether to trim the data for faster debugging. Defaults to False.

        Returns:
            DataLoader: The training data loader.
            DataLoader: The validation data loader.
        """
        dataset_size = len(self.data)
        indices = list(range(dataset_size))

        if trim:
            indices = indices[:1000]

        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
        splits = list(kf.split(indices))

        if iteration < 0 or iteration >= kfold:
            raise ValueError(f"Invalid iteration. Must be between 0 and {kfold - 1}")

        train_idx, val_idx = splits[iteration]

        train_subset = Subset(self, train_idx)
        val_subset = Subset(self, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def is_same_length(self):
        """Checks if all sequences in self.data have the same length.
        """
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


if __name__ == '__main__':
    # Load training and testing datasets for a specific RBP{i}
    sequences_file = 'data/RNAcompete_sequences.txt'
    intensities_dir = 'data/RNAcompete_intensities'
    htr_selex_dir = 'data/htr-selex'

    i = 1  # Set i to 1 for now

    train_dataset = RNASequenceDataset(sequences_file, intensities_dir, htr_selex_dir, i, train=True)

    test_dataset = RNASequenceDataset(sequences_file, intensities_dir, htr_selex_dir, i, train=False)

    train_loader, val_loader = train_dataset.create_k_fold_loaders(kfold=5, iteration=0, batch_size=32)

    # Create data loaders
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Example usage of new methods
    for batch in train_loader:
        sequences, occurrences, labels = batch
        print(sequences.shape, sequences.device)  # Should print (32, Sequence Length, Embedding Length)
