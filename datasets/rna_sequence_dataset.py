# rna_sequence_dataset.py
import torch
import pandas as pd
from datasets.base_dataset import BaseRNASequenceDataset, encode_sequence


class RNASequenceDataset(BaseRNASequenceDataset):
    def process_data(self):
        """Processes the loaded data to get maximum labels for each sequence."""
        df = pd.DataFrame(self.data)
        max_labels = df.groupby('sequence').agg({
            'occurrences': 'sum',
            'label': 'max'
        }).reset_index()
        self.data = max_labels.to_records(index=False)

    def __getitem__(self, idx):
        record = self.data[idx]
        sequence_encoded = torch.tensor(encode_sequence(record['sequence']), dtype=torch.float32)
        occurrences = torch.tensor(record['occurrences'], dtype=torch.int64)
        label = torch.tensor(record['label'], dtype=torch.int64)
        return sequence_encoded, occurrences, label

    def get_sequence_length(self):
        """Returns the length of the RNA Embeddings."""
        if self.data is not None:
            return len(self.data[0]['sequence'])
        return 0