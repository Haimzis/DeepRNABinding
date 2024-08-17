# ngram_rna_sequence_dataset.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import torch
from datasets.base_dataset import BaseRNASequenceDataset


class NgramRNASequenceDataset(BaseRNASequenceDataset):
    def __init__(self, sequences_file, intensities_dir, htr_selex_dir, i=1, trim=False, train=True, negative_examples=0, n=(4, 7), top_m="all", binary_embedding=False):
        super().__init__(sequences_file, intensities_dir, htr_selex_dir, i, trim, train, negative_examples)
        self.n = n
        self.top_m = top_m
        self.binary_embedding = binary_embedding
        self.top_indices, self.features = self.extract_and_select_top_ngrams()

    def extract_and_select_top_ngrams(self):
        """Extracts N-grams and selects the top M using chi-square scores."""
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(self.n, self.n) if isinstance(self.n, int) else self.n, binary=self.binary_embedding)
        ngram_features = vectorizer.fit_transform(self.data['sequence'])

        selector = SelectKBest(chi2, k=self.top_m)
        top_ngram_features = selector.fit_transform(ngram_features, self.data['label'])
        top_indices = selector.get_support(indices=True)

        return top_indices, top_ngram_features

    def process_data(self):
        """Processes the loaded data to filter wildcards."""
        df = pd.DataFrame(self.data)
        df = df[~df['sequence'].str.contains('N')]
        max_labels = df.groupby('sequence').agg({
            'occurrences': 'sum',
            'label': 'max'
        }).reset_index()
        self.data = max_labels.to_records(index=False)

    def __getitem__(self, idx):
        record = self.data[idx]
        features = torch.tensor(self.features[idx].toarray()[0], dtype=torch.float32)  # Ensure the tensor is on CPU
        label = torch.tensor(record['label'], dtype=torch.int64)
        return features, label

    def get_sequence_length(self):
        """Returns the length of the RNA Embeddings."""
        return self.features.shape[1] if self.features is not None else 0
