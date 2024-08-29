# main_rna.py
import sys
import os

# Add the current directory to sys.path
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import argparse
from torch.utils.data import DataLoader
from rna_sequence_dataset import RNASequenceDataset # load desired dataset 

def main():
    parser = argparse.ArgumentParser(description="Test RNASequenceDataset.")
    parser.add_argument('--sequences_file', type=str, default='data/RNAcompete_sequences_rc.txt', help='File containing the RNA sequences.')
    parser.add_argument('--intensities_dir', type=str, default='data/RNAcompete_intensities', help='Directory containing the intensity levels.')
    parser.add_argument('--htr_selex_dir', type=str, default='data/htr-selex', help='Directory containing the HTR-SELEX documents.')
    parser.add_argument('--rbp_num', type=int, default=28, help='RBP index number.')
    parser.add_argument('--trim', type=bool, default=False, help='Trim the data for faster debugging.')
    parser.add_argument('--train', type=bool, default=True, help='Load HTR-SELEX or RNACompete sequences.')
    parser.add_argument('--negative_examples', type=bool, default=False, help='Number of negative samples to generate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader.')
    args = parser.parse_args()

    dataset = RNASequenceDataset(
        sequences_file=args.sequences_file,
        intensities_dir=args.intensities_dir,
        htr_selex_dir=args.htr_selex_dir,
        num_rbp=args.rbp_num,
        trim=args.trim,
        train=args.train,
        negative_examples=args.negative_examples
    )

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f'Possible classes values: {dataset.get_possible_classes()}')
    for batch in train_loader:
        sequences, occurrences, labels = batch
        print(f'Sequences batch shape: {sequences.shape}')
        print(f'Occurrences batch shape: {occurrences.shape}')
        print(f'Labels batch shape: {labels.shape}')
        break

if __name__ == '__main__':
    main()
