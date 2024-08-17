import logging as log
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold
import torch

import args
from datasets.rna_sequence_dataset import RNASequenceDataset
from models.deepSelexCNN import DeepSELEX
from utils import save_predictions


def main(args):
    """
    Main function for the DeepSELEX model.
    """
    # Set the seed for reproducibility
    pl.seed_everything(args.seed)

    # Load the dataset
    dataset_instance = RNASequenceDataset(
        args.sequences_file, args.intensities_dir, args.htr_selex_dir, 
        args.rbp_num, train=True, trim=args.trim, negative_examples=args.negative_examples
    )

    # Initialize KFold
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    best_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_instance)):
        log.info(f"Fold {fold + 1}")

        # Create subsets for train and validation
        train_subset = torch.utils.data.Subset(dataset_instance, train_idx)
        val_subset = torch.utils.data.Subset(dataset_instance, val_idx)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Initialize the model
        model = DeepSELEX(dataset_instance.get_sequence_length(), 5, args.lr)

        # Set up early stopping
        early_stopping = EarlyStopping('val_loss', patience=args.early_stopping, verbose=True, mode='min')

        # Set up model checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=args.save_model_file,
            filename=f'best_model_fold_{fold + 1}',
            save_top_k=1,
            mode='min'
        )

        # Logger
        logger = CSVLogger(save_dir=args.log_dir, name=f'fold_{fold + 1}')

        # Initialize the trainer
        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Save the best model path
        best_model_paths.append(checkpoint_callback.best_model_path)
        log.info(f'Saved the best model for fold {fold + 1} to {checkpoint_callback.best_model_path}')

    # Use the best model from the last fold for predictions
    best_model_path = best_model_paths[-1]
    log.info(f'Using the best model from fold {len(best_model_paths)} for predictions: {best_model_path}')

    if args.predict:
        # Load the best model
        best_model = DeepSELEX.load_from_checkpoint(best_model_path)

        # Create test dataset for RBP_i
        test_dataset = RNASequenceDataset(args.sequences_file, args.intensities_dir, args.htr_selex_dir, args.rbp_num, train=False)
        test_loader = test_dataset.create_test_loader(args.batch_size)

        # Predict the intensity levels
        predictions = trainer.predict(best_model, dataloaders=test_loader)
        predictions = torch.cat(predictions, dim=0)
        save_predictions(predictions, args.predict_output_dir)
        log.info(f'Saved the predictions to {args.predict_output_dir}')

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
