import logging as log
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset
from utils import print_args, save_predictions

# Import datasets and models
from datasets.ngram_rna_sequence_dataset import NgramRNASequenceDataset
from datasets.rna_sequence_dataset import RNASequenceDataset
from datasets.rnabert_dataset import RNABERTDataset
from models.cnn_attention import CNNAttention
from models.ngrams_dnn import NGramDNN
from models.bidirectional_lstm import BiDirectionalLSTM
from models.deepselex import DeepSELEX
from models.rna_bert import RnaBert

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CNNAttention', help='Model to use: CNNAttention, DeepSELEX, LSTMSELEX, RnaBert, NGramDNN')
parser.add_argument('--rbp_num', type=int, default=1, help='The number of the RNA binding protein to predict.')
parser.add_argument('--predict', type=bool, default=True, help='Predict the intensity levels.')
parser.add_argument('--sequences_file', type=str, default='data/RNAcompete_sequences_rc.txt', help='File containing the RNA sequences.')
parser.add_argument('--intensities_dir', type=str, default='data/RNAcompete_intensities', help='Directory containing the intensity levels.')
parser.add_argument('--htr_selex_dir', type=str, default='data/htr-selex', help='Directory containing the HTR-SELEX documents.')
parser.add_argument('--predict_output_dir', type=str, default='outputs/predictions/Deep_SELEX', help='Directory to save the predictions.')
parser.add_argument('--save_model_file', type=str, default='outputs/models/Deep_SELEX', help='Directory to save the model.')
parser.add_argument('--load_model_file', type=str, default='outputs/models/Deep_SELEX/best_model.ckpt', help='File to load the model.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training.')
parser.add_argument('--early_stopping', type=int, default=10, help='Number of epochs for early stopping.')
parser.add_argument('--seed', type=int, default=24, help='Seed for random number generator.')
parser.add_argument('--trim', type=bool, default=False, help='Whether to trim the data for faster debugging.')
parser.add_argument('--negative_examples', type=bool, default=False, help='Number of negative samples to generate.')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save the logs.')
args = parser.parse_args()


def select_model_and_dataset(args):
    """
    Selects and returns the model and dataset based on provided arguments, ensuring compatibility.
    """
    # Select training and test dataset
    if args.model in ['NGramDNN']:
        train_dataset = NgramRNASequenceDataset(
            args.sequences_file, args.intensities_dir, args.htr_selex_dir,
            args.rbp_num, trim=args.trim, train=True, negative_examples=args.negative_examples, n=(7, 9), binary_embedding=False, top_m=1024
        )
        test_dataset = NgramRNASequenceDataset(
            args.sequences_file, args.intensities_dir, args.htr_selex_dir,
            args.rbp_num, trim=args.trim, train=False, negative_examples=args.negative_examples, vectorizer=train_dataset.vectorizer, selector=train_dataset.selector
        )
    elif args.model in ['CNNAttention', 'DeepSELEX', 'LSTMSELEX']:
        train_dataset = RNASequenceDataset(
            args.sequences_file, args.intensities_dir, args.htr_selex_dir,
            args.rbp_num, trim=args.trim, train=True, negative_examples=args.negative_examples
        )
        test_dataset = RNASequenceDataset(
            args.sequences_file, args.intensities_dir, args.htr_selex_dir,
            args.rbp_num, trim=args.trim, train=False, negative_examples=args.negative_examples
        )
    elif args.model in ['RnaBert']:
        train_dataset = RNABERTDataset(
            args.sequences_file, args.intensities_dir, args.htr_selex_dir,
            args.rbp_num, trim=args.trim, train=True, negative_examples=args.negative_examples, max_length=41
        )
        test_dataset = RNABERTDataset(
            args.sequences_file, args.intensities_dir, args.htr_selex_dir,
            args.rbp_num, trim=args.trim, train=False, negative_examples=args.negative_examples, max_length=41
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")

    # Select model
    if args.model == 'CNNAttention':
        model = CNNAttention(input_size=train_dataset.get_sequence_length(), num_classes=train_dataset.get_num_classes, lr=args.lr, kernel_size=9, num_filters=2048, attention_dim=512)
    elif args.model == 'DeepSELEX':
        model = DeepSELEX(input_size=train_dataset.get_sequence_length(), output_size=train_dataset.get_num_classes, lr=args.lr)
    elif args.model == 'NGramDNN':
        model = NGramDNN(input_size=train_dataset.get_sequence_length(), output_size=train_dataset.get_num_classes, lr=args.lr)
    elif args.model == 'LSTMSELEX':
        model = BiDirectionalLSTM(num_classes=train_dataset.get_num_classes, lr=args.lr)
    elif args.model == 'RnaBert':
        model = RnaBert(output_size=train_dataset.get_num_classes, lr=args.lr)
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    return model, train_dataset, test_dataset


def train_model(model, dataset, test_dataset, args):
    """
    Train the given model on the provided dataset using specified arguments.
    
    Args:
        model: PyTorch Lightning model to be trained.
        dataset: Dataset instance containing training and validation data.
        test_dataset: Dataset instance containing testing data.
        args: Namespace object containing training configurations and hyperparameters.
    """
    # Split dataset into training and validation sets
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=0.15, random_state=args.seed, shuffle=True, stratify=dataset.data['label']
    )

    # Create subsets for training and validation
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Set up early stopping
    early_stopping = EarlyStopping('val_loss', patience=args.early_stopping, verbose=True, mode='min')

    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_model_file,
        filename='best_model',
        save_top_k=1,
        mode='min'
    )

    # Set up logger
    logger = TensorBoardLogger(save_dir=args.log_dir, name='training_logs')

    # Initialize the trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        devices=[0],
        num_sanity_val_steps=0,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    log.info(f'Saved the best model to {checkpoint_callback.best_model_path}')

    if args.predict:
        # Load the best model
        best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)

        # Create a test dataset for prediction
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # Predict the intensity levels
        predictions = trainer.predict(best_model, dataloaders=test_loader)
        predictions = torch.cat(predictions, dim=0)
        save_predictions(predictions, args.predict_output_dir)

        correlation, _ = pearsonr(test_dataset.intensities, predictions)
        log.info(f'Saved the predictions to {args.predict_output_dir}')
        log.info(f'Pearson Correlation for RBP{args.rbp_num}: {correlation}')


if __name__ == '__main__':
    # Set the seed for reproducibility
    pl.seed_everything(args.seed)
    log.basicConfig(level=log.INFO)
    print_args(args)

    try:
        # Select model, training dataset, and test dataset
        model, train_dataset, test_dataset = select_model_and_dataset(args)

        # Train the model and evaluate using the test dataset
        train_model(model, train_dataset, test_dataset, args)
    except ValueError as e:
        log.error(e)
