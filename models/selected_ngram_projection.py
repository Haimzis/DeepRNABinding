import os
import numpy as np
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from datasets.ngram_rna_sequence_dataset import NgramRNASequenceDataset
from utils import transform_into_valid_intensity_range


class SelectedNGramProjectionModel:
    def __init__(self, htr_selex_dir, sequences_file, intensities_dir, KMer_LEN=(4, 7), top_k=2048, binary_embedding=False, n_jobs=-1):
        self.htr_selex_dir = htr_selex_dir
        self.sequences_file = sequences_file
        self.intensities_dir = intensities_dir
        self.kmer_length = KMer_LEN
        self.top_k = top_k  # Number of top features to select
        self.binary_embedding = binary_embedding
        self.vectorizer = None
        self.selector = None
        self.n_jobs = n_jobs

    def run(self, htr_selex_files):
        return self.process_rbp(htr_selex_files=htr_selex_files)
        
    def run_all(self, num_rbps=38):
        intensities = Parallel(n_jobs=self.n_jobs)(delayed(self.process_rbp)(rbp_id=rbp_id+1) for rbp_id in range(num_rbps))

        if self.intensities_dir:
            correlations = []
            for i, intensity in enumerate(intensities, start=1):
                binding_intensities_path = os.path.join(self.intensities_dir, f'RBP{i}.txt')
                correlations.append(self.compute_correlation(intensity, binding_intensities_path))

            return correlations
        return intensities

    def process_rbp(self, rbp_id=None, htr_selex_files=None):
        train_dataset = NgramRNASequenceDataset(
            sequences_file=self.sequences_file,
            intensities_dir=self.intensities_dir,
            htr_selex_dir=self.htr_selex_dir,
            htr_selex_files=htr_selex_files,
            rbp_num=rbp_id,
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
            htr_selex_files=htr_selex_files,
            rbp_num=rbp_id,
            trim=False,
            train=False,
            n=self.kmer_length,
            top_m=self.top_k,
            binary_embedding=self.binary_embedding,
            vectorizer=train_dataset.vectorizer,
            selector=train_dataset.selector
        )

        htr_proj_vector = train_dataset.features.sum(axis=0).A.squeeze()
        rna_scores = test_dataset.features.dot(htr_proj_vector)
        scaled_rna_scores = transform_into_valid_intensity_range(rna_scores)
        return scaled_rna_scores

    def compute_correlation(self, rna_score, binding_intensities_path):
        # Load ground truth binding intensities
        binding_intensities = np.loadtxt(binding_intensities_path)

        # Calculate Pearson correlation
        correlation, _ = pearsonr(rna_score, binding_intensities)
        return correlation
