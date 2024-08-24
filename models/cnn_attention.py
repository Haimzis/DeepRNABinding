import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CNNAttention(pl.LightningModule):
    def __init__(self, input_size, num_classes, lr=0.003, num_filters=32, kernel_size=3, attention_dim=32):
        """
        Constructor for the CNNAttentionLightning model.
        Args:
            input_size (int): The size of the input data (length of RNA sequences).
            num_classes (int): The number of output classes.
            lr (float): The learning rate for training.
            num_filters (int): The number of filters in the Conv1D layer.
            filter_size (int): The size of the convolution filter.
            attention_dim (int): The size of the attention layer.
        """
        super(CNNAttention, self).__init__()
        self.save_hyperparameters()
        
        self.conv = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size)
        self.attention = nn.Linear(num_filters, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        self.fc = nn.Linear(num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.lr = lr

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): The input data, shape (batch_size, seq_length, 4).
        Returns:
            torch.Tensor: The output data.
        """
        x = x.permute(0, 2, 1)  # Shape: (batch_size, 4, seq_length)
        conv_output = F.relu(self.conv(x))  # Shape: (batch_size, num_filters, new_seq_length)
        conv_output = conv_output.permute(0, 2, 1)  # Shape: (batch_size, new_seq_length, num_filters)

        attention_scores = torch.tanh(self.attention(conv_output))  # Shape: (batch_size, new_seq_length, attention_dim)
        attention_weights = F.softmax(self.context_vector(attention_scores), dim=1)  # Shape: (batch_size, new_seq_length, 1)

        context_vector = torch.sum(attention_weights * conv_output, dim=1)  # Shape: (batch_size, num_filters)
        output = self.fc(self.dropout(context_vector))  # Shape: (batch_size, num_classes)
        return output

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.
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
