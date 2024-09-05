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
    def __init__(self, sequences_file, intensities_dir, htr_selex_dir, htr_selex_files=None, num_rbp=None, trim=False, train=True, negative_examples=False):
        """Initializes the BaseRNASequenceDataset."""
        self.train = train
        self.sequences_file = sequences_file
        self.sequences = self.load_sequences(sequences_file)
        self.intensities_dir = intensities_dir
        self.htr_selex_dir = htr_selex_dir

        # Use init_files_list to initialize htr_selex_files
        self.htr_selex_files = self.init_files_list(htr_selex_dir, num_rbp=num_rbp, htr_selex_files=htr_selex_files)
        
        # Load intensities only if num_rbp is provided
        if num_rbp is not None:
            self.intensities = self.load_intensities(num_rbp)
        else:
            self.intensities = None
        
        self.data = self.load_data(trim, negative_examples)
        if train:
            self.process_data()

    @staticmethod
    def init_files_list(htr_selex_dir, num_rbp=None, htr_selex_files=None):
        """
        Initialize the list of htr-selex files based on either the RBP number or a list of specific files.

        Args:
            htr_selex_dir (str): Directory where htr-selex files are located.
            num_rbp (int, optional): The RBP index number. If provided, files will be loaded based on this index.
            htr_selex_files (list, optional): A list of specific htr-selex filenames to be used.

        Returns:
            list: A list of full paths to the htr-selex files.
        """
        if htr_selex_files and num_rbp:
            raise ValueError("Only one of num_rbp or htr_selex_files should be provided, not both.")

        if htr_selex_files:
            # Use the provided list of files
            file_paths = [os.path.join(htr_selex_dir, file) for file in htr_selex_files]
        elif num_rbp is not None:
            # Construct file paths using the RBP number
            file_paths = [os.path.join(htr_selex_dir, f'RBP{num_rbp}_{j}.txt') for j in range(1, 5)]
            file_paths = [file for file in file_paths if os.path.exists(file)]
        else:
            raise ValueError("Either num_rbp or htr_selex_files must be provided.")

        if not file_paths:
            raise FileNotFoundError("No valid htr-selex files found for the provided inputs.")

        return file_paths
    
    def load_sequences(self, file_path):
        """Loads RNA sequences from a file."""
        with open(file_path, 'r') as f:
            return f.read().splitlines()

    def load_intensities(self, num_rbp):
        """Loads intensities for the specified RBP index."""
        intensities_file = os.path.join(self.intensities_dir, f'RBP{num_rbp}.txt')
        if os.path.exists(intensities_file):
            return pd.read_csv(intensities_file, header=None).values.flatten()
        return None

    def load_data(self, trim, negative_examples):
        """Loads data based on whether the dataset is for training or testing."""
        if self.train:
            return self.load_train_data(trim, negative_examples)
        else:
            return self.load_test_data(trim)

    def load_train_data(self, trim, negative_examples):
        """Loads training data from htr-selex files."""
        htr_selex_files = sorted(self.htr_selex_files)
        df_list = []
        nrows = 100 if trim else None
        negative_examples = negative_examples or len(htr_selex_files) == 1

        for j, file in enumerate(htr_selex_files, start=1):
            df = pd.read_csv(file, header=None, names=['sequence', 'occurrences'], nrows=nrows)
            df['occurrences'] = df['occurrences'].astype(np.uint8)
            df['label'] = j if negative_examples else j - 1 
            df_list.append(df)

        if not df_list:
            raise FileNotFoundError("No htr-selex files found.")

        combined_df = pd.concat(df_list, ignore_index=True)

        if negative_examples:
            # Use anyway if there is a single RBP file. (binary classification)
            negative_examples_amount = len(combined_df) // len(htr_selex_files)
            neg_examples = [{'sequence': ''.join(np.random.choice(['A', 'C', 'G', 'T'], 40)),
                                'occurrences': 1, 'label': 0} for _ in range(negative_examples_amount)]
            neg_df = pd.DataFrame(neg_examples)
            combined_df = pd.concat([combined_df, neg_df], ignore_index=True)

        combined_df['sequence'] = combined_df['sequence'].astype(str)
        combined_df['label'] = combined_df['label'].astype(np.uint8)

        if trim:
            combined_df = combined_df.head(100)

        return combined_df.to_records(index=False)

    def load_test_data(self, trim):
        """Loads test data from the sequences file, with dummy labels and occurrences, and pads sequences to a length of 41."""
        df = pd.DataFrame({'sequence': self.sequences})
        df['occurrences'] = 1  # Dummy value since occurrences don't exist in test data
        df['label'] = -1  # Dummy value to signify no label

        if trim:
            df = df.head(100)

        # Pad each sequence to length 41 with 'N'
        df['sequence'] = df['sequence'].str.pad(width=41, side='right', fillchar='N')

        return df[['sequence', 'occurrences', 'label']].to_records(index=False)

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

    def get_num_classes(self):
        return len(np.unique(self.data.label))
    
    def get_possible_classes(self):
        return np.unique(self.data.label)
