import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class BiDirectionalLSTM(pl.LightningModule):
    def __init__(self, output_size, hidden_dim=128, num_layers=2, lr=0.003):
        """
        Constructor for the LSTM-based DeepSELEX model.
        Args:
            input_size (int): The length of the input sequences.
            output_size (int): The number of output classes.
            hidden_dim (int): The number of hidden units in each LSTM layer.
            num_layers (int): The number of LSTM layers.
            lr (float): The learning rate for training.
        """
        super(BiDirectionalLSTM, self).__init__()
        self.save_hyperparameters()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=4,  # Since input is one-hot encoded with 4 features per nucleotide
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # Using bidirectional LSTM
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 because of bidirectionality
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

        self.lr = lr

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output data.
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)

        # Use the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Pass through fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        """
        Configure optimizers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        Args:
            batch (tuple): A tuple containing the input data and target labels.
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The loss value for the batch.
        """
        X, _, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        Args:
            batch (tuple): A tuple containing the input data and target labels.
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The loss value for the batch.
        """
        X, _, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for the model.
        Args:
            batch (tuple): A tuple containing the input data.
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The predicted values for the batch.
        """
        X = batch[0]
        outputs = self(X)
        return torch.argmax(outputs, dim=1)