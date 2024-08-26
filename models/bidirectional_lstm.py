import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel

class BiDirectionalLSTM(BaseModel):
    def __init__(self, output_size, hidden_dim=256, num_layers=1, lr=0.003, bidirectional=False, dropout_rate=0.5):
        """
        Constructor for the LSTM-based DeepSELEX model.
        Args:
            input_size (int): The length of the input sequences.
            output_size (int): The number of output classes.
            hidden_dim (int): The number of hidden units in each LSTM layer.
            num_layers (int): The number of LSTM layers.
            lr (float): The learning rate for training.
            dropout_rate (float): Dropout rate to be applied after each fully connected layer and LSTM layers.
        """
        super(BiDirectionalLSTM, self).__init__(output_size)
        self.save_hyperparameters()

        self.bidirectional = bidirectional

        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            input_size=4,  # Since input is one-hot encoded with 4 features per nucleotide
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0  # Apply dropout only if num_layers > 1
        )

        # Adjust the input size of fc1 based on bidirectionality
        fc1_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(fc1_input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout_rate)
        
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
        lstm_out, _ = self.lstm(x)

        # Use the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Pass through fully connected layers with batch norm and dropout
        x = F.relu(self.bn1(self.fc1(lstm_out)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
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
        preds = torch.argmax(outputs, dim=1)
        self.log_metrics(loss, preds, y, 'train')
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
        preds = torch.argmax(outputs, dim=1)
        self.log_metrics(loss, preds, y, 'val')

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