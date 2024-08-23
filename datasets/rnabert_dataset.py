import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets.base_dataset import BaseRNASequenceDataset, encode_sequence
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from multimolecule import RnaTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RNABERTDataset(BaseRNASequenceDataset):
    def __init__(self, sequences_file: str, intensities_dir: str, htr_selex_dir: str, i: int = 1, train: bool = True,
                 trim: bool = False, negative_examples: int = 0, max_length: int = 41):
        super().__init__(sequences_file, intensities_dir, htr_selex_dir, i, train, trim, negative_examples)

        self.max_length = max_length
        # Initialize RNABERT tokenizer
        self.tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnabert')

    def process_data(self):
        """Processes the loaded data to get maximum labels for each sequence."""
        import pandas as pd
        df = pd.DataFrame(self.data)
        max_labels = df.groupby('sequence').agg({
            'occurrences': 'sum',
            'label': 'max'
        }).reset_index()
        self.data = max_labels.to_records(index=False)

    def __getitem__(self, idx):
        record = self.data[idx]
        # Tokenize the RNA sequence
        sequence_encoded = self.tokenizer(record['sequence'], padding='max_length', max_length=self.max_length, truncation=True,
                                return_tensors='pt')['input_ids']
        occurrences = torch.tensor(record['occurrences'], dtype=torch.int64, device=device)
        label = torch.tensor(record['label'], dtype=torch.int64, device=device)
        return sequence_encoded, occurrences, label

    @staticmethod
    def collate_fn(batch):
        sequences, occurrences, labels = zip(*batch)
        return torch.stack(sequences), torch.stack(occurrences), torch.stack(labels)

    def create_test_loader(self, batch_size: int = 32) -> DataLoader:
        """Creates a test data loader for the dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=RNABERTDataset.collate_fn)