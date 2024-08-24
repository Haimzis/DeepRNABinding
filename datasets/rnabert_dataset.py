import torch
from datasets.base_dataset import BaseRNASequenceDataset

import torch
from multimolecule import RnaTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RNABERTDataset(BaseRNASequenceDataset):
    def __init__(self, sequences_file: str, intensities_dir: str, htr_selex_dir: str, i: int = 1, train: bool = True,
                 trim: bool = False, negative_examples: bool = False, max_length: int = 440):
        super().__init__(sequences_file, intensities_dir, htr_selex_dir, i, trim, train, negative_examples)

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
        # Tokenize the RNA sequence and include attention_mask
        encoded_sequence = self.tokenizer(
            record['sequence'],
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_sequence['input_ids'].squeeze(0)  # Shape (max_length,)
        attention_mask = encoded_sequence['attention_mask'].squeeze(0)  # Shape (max_length,)
        
        occurrences = torch.tensor(record['occurrences'], dtype=torch.int64)
        label = torch.tensor(record['label'], dtype=torch.int64)

        return input_ids, attention_mask, occurrences, label


