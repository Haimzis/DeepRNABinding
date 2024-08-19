import os
import numpy as np
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import pearsonr

class BaselineModel:
    def __init__(self, htr_selex_dir, sequences_file, intensities_dir, KMer_LEN=7):
        self.htr_selex_dir = htr_selex_dir
        self.sequences_file = sequences_file
        self.intensities_dir = intensities_dir
        self.kmer_length = KMer_LEN

    def compute_baseline(self, num_rbps=38):
        correlations = []
        for i in range(num_rbps):
            last_cycle_path = os.path.join(self.htr_selex_dir, f'RBP{i + 1}_4.txt')
            if not os.path.exists(last_cycle_path):
                last_cycle_path = os.path.join(self.htr_selex_dir, f'RBP{i + 1}_3.txt')
            rnacompete_path = self.sequences_file
            binding_intensities_path = os.path.join(self.intensities_dir, f'RBP{i + 1}.txt')
            correlation = self.compute_baseline_helper(last_cycle_path, rnacompete_path, binding_intensities_path)
            correlations.append(correlation)
            print(f"Correlation for RBP{i + 1}: {correlation}")

        with open('baseline_correlations.txt', 'w') as file:
            for correlation in correlations:
                file.write(f'{correlation}\n')

        print(f"Correlations saved to 'baseline_correlations.txt'")
        return correlations

    def compute_baseline_helper(self, last_cycle_path, rnacompete_path, binding_intensities_path):
        all_7mers = [''.join(combo) for combo in product('ACGT', repeat=self.kmer_length)]
        vectorizer = CountVectorizer(vocabulary=all_7mers, analyzer='char', ngram_range=(self.kmer_length, self.kmer_length), lowercase=False)

        with open(last_cycle_path, 'r') as file:
            sequences = [line.strip().split(',')[0] for line in file if 'N' not in line]

        counts = vectorizer.fit_transform(sequences)
        seven_mer_scores = np.array(counts.sum(axis=0)).flatten()

        with open(rnacompete_path, 'r') as file:
            rnacompete_sequences = [line.strip().split()[0] for line in file]

        rnacompete_counts = vectorizer.fit_transform(rnacompete_sequences)
        rnacompete_scores = rnacompete_counts.dot(seven_mer_scores)
        seq_lens = np.array([len(seq) - self.kmer_length + 1 for seq in rnacompete_sequences])
        rnacompete_scores = rnacompete_scores / seq_lens

        binding_intensities = np.loadtxt(binding_intensities_path)
        correlation, _ = pearsonr(rnacompete_scores, binding_intensities)
        return correlation
