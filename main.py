import logging as log
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

import args
import dataset
from models.deepSelexCNN import DeepSELEX
from utils import save_predictions


def main(args):
    """
    Main function for the DeepSELEX model.
    """
    # Set the seed for reproducibility
    pl.seed_everything(args.seed)

    # Load the training dataset
    train_dataset = dataset.RNASequenceDataset(args.sequences_file, args.intensities_dir, args.htr_selex_dir, args.rbp_num, train=True)

    # Create k-fold data loaders
    # TODO: Fix num_workers - need to add worker_init_func, Haim: you can leave it to me.
    #       Use larger batch size and epochs as possible.
    train_loader, val_loader = train_dataset.create_k_fold_loaders(args.kfold, 0, args.batch_size, trim=True, negative_examples=0, num_workers=0)

    # Initialize the model
    model = DeepSELEX(train_dataset.get_sequence_length(), 5, args.lr)

    # Set up early stopping
    # TODO: Use better metric for earlystopping - for example, val_accuracy.
    early_stopping = EarlyStopping('val_loss', patience=args.early_stopping, verbose=True)
    
    # Set up model checkpointing
    # TODO: Use better metric for checkpoint - for example, val_accuracy.
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_model_file,
        filename='best_model',
        save_top_k=1,
        mode='min'
    )

    # Initialize the trainer
    trainer = Trainer(
        max_epochs=args.epochs, 
        callbacks=[early_stopping, checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the best model
    best_model_path = checkpoint_callback.best_model_path
    log.info(f'Saved the best model to {best_model_path}')

    if args.predict:
        # Load the best model
        best_model = DeepSELEX.load_from_checkpoint(best_model_path)

        # Create test dataset for RBP_i
        test_dataset = dataset.RNASequenceDataset(args.sequences_file, args.intensities_dir, args.htr_selex_dir, args.rbp_num, train=False)
        test_loader = test_dataset.create_test_loader(args.batch_size, include_targets=False)

        # Predict the intensity levels
        predictions = trainer.predict(best_model, dataloaders=test_loader)
        predictions = torch.cat(predictions, dim=0)
        save_predictions(predictions, args.predict_output_dir)
        log.info(f'Saved the predictions to {args.predict_output_dir}')
        

if __name__ == '__main__':
    args = args.parse_args()
    log.basicConfig(level=log.INFO)
    main(args)
