import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ImprovedDeepSELEX(pl.LightningModule):
    def __init__(self, input_size, output_size, lr=0.001, dropout_rate=0.2):
        """
        Constructor for the Improved DeepSELEX model.
        Args:
            input_size (int): The size of the input data.
            output_size (int): The size of the output data.
            lr (float): The learning rate for training.
            dropout_rate (float): The dropout rate for regularization.
        """
        super(ImprovedDeepSELEX, self).__init__()
        self.save_hyperparameters()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=8, stride=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=8, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        
        # Calculate the size of the flattened output from the convolutional layers
        self.flat_size = self._get_conv_output_size(input_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        self.lr = lr

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
            dummy_output = self.pool(self.conv2(self.pool(self.conv1(dummy_input))))
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
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

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
        accuracy = (outputs.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
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
        return torch.softmax(outputs, dim=1)
