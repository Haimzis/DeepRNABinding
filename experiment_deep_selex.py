"""
This module is the solution DeepSELEX represented in class for the assignment.
The module structure is as follows:
    1. Input: DNA sequences represented as One-Hot Encoded matrix.
    2. Convolutional Layer:
        - 512 filters
        - Kernel size of 8
        - stride of 1
    3. Max Pooling Layer:
        - Kernel size of 5
        - stride of 5
    4. 3 Fully Connected Layers:
        - activation function: ReLU
        - hidden units: 64, 32, 32
        - output units: 5 (representing the intensity levels)

The dataset usage is as follows:
--------------------------------
There are three kind of files:
    1. RNAcompete_sequences.txt
        - Contains the RNA sequences to predict the intensity levels.
    2. RNAcompete_intensities
        - Contains the intensity levels for the RNA sequences for each RBP{i}.
        - There are "RBP1.txt",..., "RBP38.txt" files which will be used to train.
    3. htr-selex
        - Contains the documents for the RBP{i} stating which RNA sequences
          survived the selection process in each round.

Our goal is to predict the intensity levels for "RBP39.txt",..., "RBP76.txt" files.


"""
import argparse
import logging as log
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import utils
import dataset

class DeepSELEX(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Constructor for the DeepSELEX model.
        Args:
            input_size (int): The size of the input data.
            output_size (int): The size of the output data.
        """
        super(DeepSELEX, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=512, kernel_size=8, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)

        # Calculate the size of the flattened output from the convolutional layers
        self.flat_size = self._get_conv_output_size(input_size)

        self.fc1 = nn.Linear(self.flat_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, output_size)

    def _get_conv_output_size(self, seq_length):
        """
        Calculate the size of the flattened output from the convolutional layers.
        Args:
            seq_length (int): The length of the input sequence.
        Returns:
            int: The size of the flattened output.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, seq_length)
            dummy_output = self.pool(self.conv1(dummy_input))
            return dummy_output.numel()
    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output data.
        """
        # Transpose the input: from (batch, seq_length, channels) to (batch, channels, seq_length)
        x = x.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def loss(self, outputs, targets):
        """
        Calculate the loss of the model.
        Args:
            outputs (torch.Tensor): The output data.
            targets (torch.Tensor): The target data.
        Returns:
            torch.Tensor: The loss value.
        """
        return F.cross_entropy(outputs, targets)

def main(parsed_args):
    """
    Main function for the DeepSELEX model.
    """
    best_model = None
    best_loss = float('inf')

    # Set the seed for reproducibility
    if not parsed_args.seed:
        seed = random.randint(0, 2**31)
    utils.set_seed(parsed_args.seed)

    # Set the device for computation
    device = utils.get_device()

    # Train and predict the model on RBP_i
    if parsed_args.train:
        train_dataset = dataset.RNASequenceDataset(parsed_args.sequences_file, parsed_args.intensities_dir, parsed_args.htr_selex_dir, 1, train=True)
        # do cross validation on RBP1
        for i in range(parsed_args.kfold):
            train_loader, val_loader = train_dataset.create_k_fold_loaders(parsed_args.kfold, i, parsed_args.batch_size, True)
            model = DeepSELEX(train_dataset.get_sequence_length(), 5)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=parsed_args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
            torch_trainer = utils.TorchTrainer(model=model, optimizer=optimizer, scheduler=None, device=device)
            torch_trainer.fit(train_loader, val_loader, num_epochs=parsed_args.epochs, early_stopping=parsed_args.early_stopping)
            if torch_trainer.best_val_loss < best_loss:
                best_loss = torch_trainer.best_val_loss
                best_model = model
        # Save the best model


def parse_cli():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='DeepSELEX model for predicting intensity levels.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode.')
    parser.add_argument('--train', action='store_true',
                        help='Train the model.')
    parser.add_argument('--predict', action='store_true',
                        help='Predict the intensity levels.')
    parser.add_argument('--log', action='store_true',
                        help='Enable logging.')
    parser.add_argument('--log_dir', type=str, default='Logs',
                        help='Directory to save logs.')
    parser.add_argument('--sequences_file', type=str, default='data/RNAcompete_sequences.txt',
                        help='File containing the RNA sequences.')
    parser.add_argument('--intensities_dir', type=str, default='data/RNAcompete_intensities',
                        help='Directory containing the intensity levels.')
    parser.add_argument('--htr_selex_dir', type=str, default='data/htr-selex',
                        help='Directory containing the HTR-SELEX documents.')
    parser.add_argument('--predict_output_dir', type=str, default='outputs/predictions/Deep_SELEX',
                        help='Directory to save the predictions.')
    parser.add_argument('--save_model_file', type=str, default='outputs/models/Deep_SELEX.pth',
                        help='File to save the model.')
    parser.add_argument('--load_model_file', type=str, default='outputs/models/Deep_SELEX.pth',
                        help='File to load the model.')
    parser.add_argument('--dev_ratio', type=float, default=0.1,
                        help='Ratio of the dataset to use for development.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='Learning rate for training.')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs for early stopping.')
    parser.add_argument('--seed', type=int, default=23,
                        help='Seed for random number generator.')
    return parser.parse_args()

if __name__ == '__main__':
    if utils.is_debugging():
        parsed_args = argparse.Namespace(
            debug=True,
            train=True,
            predict=True,
            log=True,
            log_dir='Logs',
            sequences_file='data/RNAcompete_sequences.txt',
            intensities_dir = 'data/RNAcompete_intensities',
            htr_selex_dir = 'data/htr-selex',
            predict_output_dir=f'outputs/predictions/Deep_SELEX',
            save_model_file=f'outputs/models/Deep_SELEX.pth',
            load_model_file=f'outputs/models/Deep_SELEX.pth',
            kfold=10,
            batch_size=32,
            epochs=25,
            lr=0.01,
            early_stopping=10,
            seed=23,
        )
    else:
        parsed_args = parse_cli()
    log.basicConfig(level=log.INFO)
    main(parsed_args)


