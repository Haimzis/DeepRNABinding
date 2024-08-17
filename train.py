import argparse
import logging as log

import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Set the multiprocessing start method
mp.set_start_method('spawn', force=True)

import args
# import dataset
# from datasets.ngram_rna_sequence_dataset import NgramRNASequenceDataset
from datasets.rna_sequence_dataset import RNASequenceDataset
# from models.deepSelexDNN import DeepSELEX
from models.deepSelexCNN import DeepSELEX
from utils import save_predictions
from itertools import product


def main(args):
    """
    Main function for the DeepSELEX model.
    """
    # Set the seed for reproducibility
    pl.seed_everything(args.seed)

    # Load the dataset
    # dataset_instance = NgramRNASequenceDataset(
    #     args.sequences_file, args.intensities_dir, args.htr_selex_dir,
    #     args.rbp_num, trim=args.trim, train=True, negative_examples=args.negative_examples
    # )
    dataset_instance = RNASequenceDataset(
        args.sequences_file, args.intensities_dir, args.htr_selex_dir,
        args.rbp_num, trim=args.trim, train=True, negative_examples=args.negative_examples
    )

    # Split dataset into training and validation
    train_idx, val_idx = train_test_split(
        range(len(dataset_instance)), test_size=0.05, random_state=args.seed, shuffle=True, stratify=dataset_instance.data['label']
    )

    # Create subsets for train and validation
    train_subset = torch.utils.data.Subset(dataset_instance, train_idx)
    val_subset = torch.utils.data.Subset(dataset_instance, val_idx)

    # Define hyperparameter space
    lr_values = [0.005]
    batch_size_values = [256]
    hyperparameter_combinations = list(product(lr_values, batch_size_values))

    best_val_loss = float('inf')
    best_model_path = None

    for lr, batch_size in hyperparameter_combinations:
        log.info(f"Testing combination: lr={lr}, batch_size={batch_size}")

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

        # Initialize the model
        model = DeepSELEX(dataset_instance.get_sequence_length(), 4, lr)

        # Set up early stopping
        early_stopping = EarlyStopping('val_loss', patience=args.early_stopping, verbose=True, mode='min')

        # Set up model checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=args.save_model_file,
            filename=f'best_model_lr_{lr}_batch_size_{batch_size}',
            save_top_k=1,
            mode='min'
        )

        # Logger
        # logger = CSVLogger(save_dir=args.log_dir, name=f'lr_{lr}_batch_size_{batch_size}')
        logger = TensorBoardLogger(save_dir=args.log_dir, name=f'lr_{lr}_batch_size_{batch_size}')

        # Initialize the trainer
        trainer = Trainer(
            max_epochs=1, #args.epochs,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
            devices=[0],
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Check if this model has the best validation loss
        if checkpoint_callback.best_model_score < best_val_loss:
            best_val_loss = checkpoint_callback.best_model_score
            best_model_path = checkpoint_callback.best_model_path

        log.info(f'Saved the best model for combination lr={lr}, batch_size={batch_size} to {checkpoint_callback.best_model_path}')

    log.info(f'Using the best model for predictions: {best_model_path}')

    if args.predict:
        # Load the best model
        best_model = DeepSELEX.load_from_checkpoint(best_model_path)

        # Create test dataset for RBP_i
        # test_dataset = NgramRNASequenceDataset(args.sequences_file, args.intensities_dir, args.htr_selex_dir, args.rbp_num, train=False)
        test_dataset = RNASequenceDataset(args.sequences_file, args.intensities_dir, args.htr_selex_dir, trim=args.trim, train=False)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

        # Predict the intensity levels
        predictions = trainer.predict(best_model, dataloaders=test_loader)
        predictions = torch.cat(predictions, dim=0)
        save_predictions(predictions, args.predict_output_dir)
        correlation, _ = pearsonr(test_dataset.intensities, predictions)
        log.info(f'Saved the predictions to {args.predict_output_dir}')
        log.info(f'Pearson Correlation for RBP{args.rbp_num}: {correlation}')



if __name__ == '__main__':
    args = args.parse_args()
    '''args = argparse.Namespace(rbp_num=1,
                                predict=True,
                                sequences_file='data/RNAcompete_sequences.txt',
                                intensities_dir='data/RNAcompete_intensities',
                                htr_selex_dir='data/htr-selex',
                                predict_output_dir='outputs/predictions/Deep_SELEX',
                                save_model_file='outputs/models/Deep_SELEX.pth',
                                load_model_file='outputs/models/Deep_SELEX.pth',
                                batch_size=64,
                                epochs=100,
                                lr=0.001,
                                early_stopping=10,
                                seed=32,
                                kfold=10,
                                trim=False,
                                negative_examples=1000,
                                log_dir='outputs/logs/Deep_SELEX')'''
    log.basicConfig(level=log.INFO)
    main(args)
