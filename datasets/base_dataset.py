import os
from collections import defaultdict
from random import random, choices, choice

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
negative_examples_gen = {'RANDOM': 0, 'FROM_CYCLE_1': 1, 'MARKOV': 2}

def encode_sequence(sequence):
    """Encodes a nucleotide sequence into a one-hot encoded format."""
    try:
        return [nucleotide_to_onehot[nuc] for nuc in sequence]
    except KeyError:
        raise KeyError(f"Unknown nucleotide encountered in sequence: {sequence}")

class BaseRNASequenceDataset(Dataset):
    def __init__(self, sequences_file, intensities_dir, htr_selex_dir, i=1, trim=False, train=True, negative_examples=False):
        """Initializes the BaseRNASequenceDataset."""
        self.train = train
        self.sequences_file = sequences_file
        self.sequences = self.load_sequences(sequences_file)
        self.intensities_dir = intensities_dir
        self.htr_selex_dir = htr_selex_dir
        self.i = i
        self.intensities = self.load_intensities()
        self.data = self.load_data(trim, negative_examples)
        if train:
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
        """Loads data based on whether the dataset is for training or testing."""
        if self.train:
            return self.load_train_data(trim, negative_examples)
        else:
            return self.load_test_data(trim)

    def load_train_data(self, trim, negative_examples, gen_example_choice='RANDOM'):
        """Loads training data from htr-selex files for the specified RBP index."""
        htr_selex_files = [os.path.join(self.htr_selex_dir, f'RBP{self.i}_{j}.txt') for j in range(1, 5)]
        htr_selex_files = [file for file in htr_selex_files if os.path.exists(file)]
        df_list = []
        nrows = 100 if trim else None
        negative_examples = negative_examples or len(htr_selex_files) == 1

        for j, file in enumerate(htr_selex_files, start=1):
            df = pd.read_csv(file, header=None, names=['sequence', 'occurrences'], nrows=nrows)
            df['occurrences'] = df['occurrences'].astype(np.uint8)
            df['label'] = j if negative_examples else j - 1 
            df_list.append(df)

        if not df_list:
            raise FileNotFoundError(f"No htr-selex files found for RBP{self.i}")

        combined_df = pd.concat(df_list, ignore_index=True)

        if negative_examples:
            # Use anyway if there is a single RBP file. (binary classification)
            negative_examples_amount = len(combined_df) // len(htr_selex_files)
            if gen_example_choice == 'RANDOM':
                neg_examples = [{'sequence': ''.join(np.random.choice(['A', 'C', 'G', 'T'], 40)),
                                 'occurrences': 1, 'label': 0} for _ in range(negative_examples_amount)]
                neg_df = pd.DataFrame(neg_examples)
            elif gen_example_choice == 'FROM_CYCLE_1':
                neg_df = pd.DataFrame(self.get_negative_examples_by_other_cycle_1(negative_examples_amount))
            elif gen_example_choice == 'MARKOV':
                neg_df = pd.DataFrame(self.get_negative_examples_by_markov_chain(negative_examples_amount))
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

    def get_negative_examples_by_other_cycle_1(self, negative_examples):
        """
        The First method suggested to generate negative examples:
        In a case where RBP{self.i}_1.txt in htr-selex_dir is missing, this function will return negative examples
        based on all other sequences of RBP{j}_1.txt where j != self.i.
        this method sample #negative_examples from the sequences of RBP{j}_1.txt where j != self.i.
        Returns: a list of negative examples
        """
        if not hasattr(self, 'cycle_1_sequences'):
            self.load_cycle_1_sequences()

        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(self.cycle_1_sequences)

        # Unpack sequences and occurrences
        sequences = df['sequence'].values
        occurrences = df['occurrences'].values

        # Normalize the occurrences to avoid overflow
        total_occurrences = sum(occurrences)
        normalized_weights = [occ / total_occurrences for occ in occurrences]

        # Sample sequences based on normalized weights
        sampled_indices = choices(range(len(sequences)), weights=normalized_weights, k=negative_examples)
        sampled_sequences = [sequences[i] for i in sampled_indices]

        # Create a structured array
        neg_examples = np.array([(seq, 1, 0) for seq in sampled_sequences],
                                dtype=[('sequence', 'U40'), ('occurrences', 'u1'), ('label', 'u1')])

        return neg_examples

    def load_cycle_1_sequences(self):
        """Load sequences from Cycle 1 files and aggregate occurrences and labels."""
        self.cycle_1_sequences = []
        htr_selex_files = [os.path.join(self.htr_selex_dir, f'RBP{i}_1.txt') for i in range(1, 39)]
        htr_selex_files = [file for file in htr_selex_files if os.path.exists(file)]
        df_list = []

        for file in htr_selex_files:
            df = pd.read_csv(file, header=None, names=['sequence', 'occurrences'])
            df['occurrences'] = df['occurrences'].astype(np.uint8)
            df['label'] = 1
            df_list.append(df)

        if not df_list:
            raise FileNotFoundError("No sequences found in htr-selex files")

        combined_df = pd.concat(df_list, ignore_index=True)

        # Group by sequence and aggregate occurrences and max label
        max_labels = combined_df.groupby('sequence').agg({
            'occurrences': 'sum',
            'label': 'max'
        }).reset_index()

        # Convert to records format and store in cycle_1_sequences
        self.cycle_1_sequences = max_labels.to_records(index=False)

    def get_negative_examples_by_markov_chain(self, negative_examples, order=1):
        """
        Generate negative examples using a highly optimized Markov chain model.

        Args:
        negative_examples (int): Number of negative examples to generate.
        order (int): The order of the Markov chain. Default is 1.

        Returns:
        numpy.recarray: A record array of negative examples.
        """
        if not hasattr(self, 'cycle_1_sequences'):
            self.load_cycle_1_sequences()
        if not hasattr(self, 'markov_chain'):
            self.build_markov_chain(order)

        # Generate new sequences
        new_sequences = []
        for _ in range(negative_examples):
            new_seq = ''.join(choice('ACGT') for _ in range(order))
            while len(new_seq) < 40:  # Assuming sequences are 40 nucleotides long
                state = new_seq[-order:] if len(new_seq) >= order else new_seq
                next_char = self.markov_chain[state].generate()
                new_seq += next_char

            new_sequences.append(new_seq)

        # Create a structured array directly
        return np.array([(seq, 1, 0) for seq in new_sequences],
                        dtype=[('sequence', 'U40'), ('occurrences', 'u1'), ('label', 'u1')])

    def build_markov_chain(self, order):
        """
        Build and cache the Markov chain.
        """
        chain = defaultdict(lambda: MarkovState())
        for seq in self.cycle_1_sequences['sequence']:
            for i in range(len(seq) - order):
                state = seq[i:i + order]
                next_char = seq[i + order]
                chain[state].add(next_char)
        self.markov_chain = chain

class MarkovState:
    def __init__(self):
        self.counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        self.total = 0

    def add(self, char):
        if char == 'N':
            return
        self.counts[char] += 1
        self.total += 1

    def generate(self):
        if self.total == 0:
            return choice('ACGT')  # Use fallback directly if total is zero
        r = random() * self.total
        for char, count in self.counts.items():
            r -= count
            if r <= 0:
                return char
        return choice('ACGT')  # Fallback, should rarely happen