import os
import numpy as np
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from datasets.ngram_rna_sequence_dataset import NgramRNASequenceDataset
import optuna

class NgramModel:
    def __init__(self, htr_selex_dir, sequences_file, intensities_dir, KMer_LEN=(4, 7), top_k=2048, binary_embedding=False, trial=None, n_jobs=-1):
        self.htr_selex_dir = htr_selex_dir
        self.sequences_file = sequences_file
        self.intensities_dir = intensities_dir
        self.kmer_length = KMer_LEN
        self.top_k = top_k  # Number of top features to select
        self.binary_embedding = binary_embedding
        self.vectorizer = None
        self.selector = None
        self.trial = trial
        self.n_jobs = n_jobs

    def run(self, num_rbps=38):
        correlations = Parallel(n_jobs=self.n_jobs)(delayed(self.process_rbp)(i) for i in range(num_rbps))

        with open('ngram_correlations.txt', 'w') as file:
            for correlation in correlations:
                file.write(f'{correlation}\n')

        print(f"Correlations saved to 'ngram_correlations.txt'")
        return correlations

    def process_rbp(self, i):
        binding_intensities_path = os.path.join(self.intensities_dir, f'RBP{i + 1}.txt')

        train_dataset = NgramRNASequenceDataset(
            sequences_file=self.sequences_file,
            intensities_dir=self.intensities_dir,
            htr_selex_dir=self.htr_selex_dir,
            i=i + 1,
            trim=False,
            train=True,
            n=self.kmer_length,
            top_m=self.top_k,
            binary_embedding=self.binary_embedding
        )

        test_dataset = NgramRNASequenceDataset(
            sequences_file=self.sequences_file,
            intensities_dir=self.intensities_dir,
            htr_selex_dir=self.htr_selex_dir,
            i=i + 1,
            trim=False,
            train=False,
            n=self.kmer_length,
            top_m=self.top_k,
            binary_embedding=self.binary_embedding,
            vectorizer=train_dataset.vectorizer,
            selector=train_dataset.selector
        )

        htr_proj_vector = train_dataset.features.sum(axis=0).A.squeeze()
        rna_score = test_dataset.features.dot(htr_proj_vector)
        correlation = self.compute_correlation(rna_score, binding_intensities_path)

        if self.trial is not None:
            self.trial.report(np.array(correlation), i)

            if self.trial.should_prune():
                raise optuna.TrialPruned()

        print(f"Correlation for RBP{i + 1}: {correlation}")
        return correlation

    def compute_correlation(self, rna_score, binding_intensities_path):
        # Load ground truth binding intensities
        binding_intensities = np.loadtxt(binding_intensities_path)

        # Calculate Pearson correlation
        correlation, _ = pearsonr(rna_score, binding_intensities)
        return correlation
