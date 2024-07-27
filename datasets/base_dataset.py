# base_dataset.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nucleotide_to_onehot = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0.25, 0.25, 0.25, 0.25]
}

def encode_sequence(sequence):
    """Encodes a nucleotide sequence into a one-hot encoded format."""
    try:
        return [nucleotide_to_onehot[nuc] for nuc in sequence]
    except KeyError:
        raise KeyError(f"Unknown nucleotide encountered in sequence: {sequence}")

class BaseRNASequenceDataset(Dataset):
    def __init__(self, sequences_file, intensities_dir, htr_selex_dir, i=1, train=True, trim=False, negative_examples=0):
        """Initializes the BaseRNASequenceDataset."""
        self.sequences = self.load_sequences(sequences_file)
        self.intensities_dir = intensities_dir
        self.htr_selex_dir = htr_selex_dir
        self.i = i
        self.train = train
        self.intensities = self.load_intensities()
        self.data = self.load_data(trim, negative_examples)
        self.process_data()

    def load_sequences(self, file_path):
        """Loads RNA sequences from a file."""
        with open(file_path, 'r') as f:
            return f.read().splitlines()

    def load_intensities(self):
        """Loads intensities for the specified RBP index."""
        intensities_file = os.path.join(self.intensities_dir, f'RBP{self.i}.txt')
        if os.path.exists(intensities_file):
            return pd.read_csv(intensities_file, header=None).values.flatten()
        return None

    def load_data(self, trim, negative_examples):
        """Loads data from htr-selex files for the specified RBP index."""
        htr_selex_files = [os.path.join(self.htr_selex_dir, f'RBP{self.i}_{j}.txt') for j in range(1, 5)]
        df_list = []
        nrows = 100 if trim else None

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

        return combined_df.to_records(index=False)

    def process_data(self):
        """Processes the loaded data."""
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_sequence_length(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_intensities(self):
        """Returns the array of intensity values."""
        return self.intensities

    def get_sequences(self):
        """Returns the list of RNA sequences."""
        return self.sequences
    
    def is_same_length(self):
        """Checks if all sequences in self.data have the same length."""
        return len(set(len(record['sequence']) for record in self.data)) == 1