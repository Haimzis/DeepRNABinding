# rna_sequence_dataset.py
import torch
import pandas as pd
from datasets.rna_sequence_dataset import RNASequenceDataset, encode_sequence


class RNASequenceDatasetDeepSelex(RNASequenceDataset):
    def __init__(self, sequences_file, intensities_dir, htr_selex_dir, htr_selex_files=None, num_rbp=None, trim=False, train=True, negative_examples=False, k=14):
        super().__init__(sequences_file, intensities_dir, htr_selex_dir, htr_selex_files, num_rbp, trim, train, negative_examples)
        self.k = k

    def __getitem__(self, idx):
        record = self.data[idx]
        # Encode the sequence and pad it with 4 'N' from each side
        if len(record['sequence']) < self.k:
            # Pad the sequence by (k-L)/2 uniform vectors (N) on each side
            pad_str = 'N' * ((self.k - len(record['sequence'])) // 2)
        else:
            # Pad the sequence by 4 uniform vectors (N) on each side
            pad_str = 'N' * 4
        sequence_encoded = torch.tensor(encode_sequence(pad_str + record['sequence'] + pad_str), dtype=torch.float32)
        # divide the sequence into k-mers efficiently
        # the seq is padded so now L>=k
        sequence_encoded = sequence_encoded.unfold(0, self.k, 1).permute(0, 2, 1)
        occurrences = torch.tensor(record['occurrences'], dtype=torch.int64)
        label = torch.tensor(record['label'], dtype=torch.int64)
        return sequence_encoded, occurrences, label

    def get_sequence_length(self):
        """Returns the length of the RNA Embeddings."""
        if self.data is not None:
            res = len(self.data[0]['sequence'])
            if res < self.k:
                return self.k
            return res + 8
        return 0

    def get_k(self):
        return self.k