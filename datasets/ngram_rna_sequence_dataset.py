# ngram_rna_sequence_dataset.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import torch
from datasets.base_dataset import BaseRNASequenceDataset


class NgramRNASequenceDataset(BaseRNASequenceDataset):
    def __init__(self, sequences_file, intensities_dir, htr_selex_dir, i=1, trim=False, train=True, negative_examples=False, n=(4, 7), top_m=2048, binary_embedding=False, vectorizer=None, selector=None):
        super().__init__(sequences_file, intensities_dir, htr_selex_dir, i, trim, train, negative_examples)
        self.n = n
        self.top_m = top_m
        self.binary_embedding = binary_embedding
        self.vectorizer = vectorizer
        self.selector = selector
        self.top_indices, self.features = self.extract_and_select_top_ngrams()

    def extract_and_select_top_ngrams(self):
        """Extracts N-grams and selects the top M using chi-square scores."""
        if self.vectorizer is None:
            # Create vectorizer in the training mode
            self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(self.n, self.n) if isinstance(self.n, int) else self.n, binary=self.binary_embedding)
        
        ngram_features = self.vectorizer.fit_transform(self.data['sequence']) if self.train else self.vectorizer.transform(self.data['sequence'])

        if self.train:
            # Only fit the selector in training mode
            self.selector = SelectKBest(chi2, k=self.top_m)
            top_ngram_features = self.selector.fit_transform(ngram_features, self.data['label'])
            top_indices = self.selector.get_support(indices=True)
        else:
            # Use the already-fitted selector in test mode
            if self.selector is None:
                raise ValueError("Selector must be provided in test mode.")
            top_ngram_features = self.selector.transform(ngram_features)
            top_indices = None  # Not needed in test mode

        return top_indices, top_ngram_features

    def process_data(self):
        """Processes the loaded data to filter wildcards and handle train/test modes."""
        df = pd.DataFrame(self.data)
        df = df[~df['sequence'].str.contains('N')]  # Remove sequences with 'N'
        
        if self.train:
            max_labels = df.groupby('sequence').agg({
                'occurrences': 'sum',
                'label': 'max'
            }).reset_index()
            self.data = max_labels.to_records(index=False)
        else:
            df['label'] = -1  # Dummy label for test data
            self.data = df.to_records(index=False)

    def __getitem__(self, idx):
        record = self.data[idx]
        features = torch.tensor(self.features[idx].toarray()[0], dtype=torch.float32)  # Ensure the tensor is on CPU
        label = torch.tensor(record['label'], dtype=torch.int64)
        return features, label

    def get_sequence_length(self):
        """Returns the length of the RNA Embeddings."""
        return self.features.shape[1] if self.features is not None else 0
