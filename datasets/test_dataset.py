# main_rna.py
import argparse
from torch.utils.data import DataLoader
from rna_sequence_dataset import RNASequenceDataset # load desired dataset 

def main():
    parser = argparse.ArgumentParser(description="Test RNASequenceDataset.")
    parser.add_argument('--sequences_file', type=str, required=True, help='Path to the RNA sequences file.')
    parser.add_argument('--intensities_dir', type=str, required=True, help='Directory containing intensity files.')
    parser.add_argument('--htr_selex_dir', type=str, required=True, help='Directory containing htr-selex files.')
    parser.add_argument('--rbp_num', type=int, required=True, help='RBP index number.')
    parser.add_argument('--trim', type=bool, default=False, help='Trim the data for faster debugging.')
    parser.add_argument('--negative_examples', type=int, default=0, help='Number of negative samples to generate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader.')
    args = parser.parse_args()

    dataset = RNASequenceDataset(
        sequences_file=args.sequences_file,
        intensities_dir=args.intensities_dir,
        htr_selex_dir=args.htr_selex_dir,
        i=args.rbp_num,
        train=True,
        trim=args.trim,
        negative_examples=args.negative_examples
    )

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for batch in train_loader:
        sequences, occurrences, labels = batch
        print(f'Sequences batch shape: {sequences.shape}')
        print(f'Occurrences batch shape: {occurrences.shape}')
        print(f'Labels batch shape: {labels.shape}')

if __name__ == '__main__':
    main()
