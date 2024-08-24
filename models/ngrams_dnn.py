import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel

class NGramDNN(BaseModel):
    def __init__(self, input_size, output_size, lr=0.003, dropout_rate=0.25):
        """
        Constructor for the DeepSELEXDNN model adapted for N-gram features.
        Args:
            input_size (int): The size of the input data (number of N-gram features).
            output_size (int): The size of the output data (number of classes or regression targets).
            lr (float): The learning rate for training.
            dropout_rate (float): The dropout rate for regularization.
        """
        super(NGramDNN, self).__init__(output_size)
        self.save_hyperparameters()

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.lr = lr

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output data.
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # Output layer, apply softmax in the loss function for classification
        return x

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
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
        X, y = batch  # Assuming X is (N, M) and y is (N,)
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
        X, y = batch
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