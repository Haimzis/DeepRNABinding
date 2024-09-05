import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel

class DeepSELEX(BaseModel):
    def __init__(self, seq_size, k, output_size, lr=0.001):
        """
        Constructor for the DeepSELEX model.
        Args:
            input_size (int): The size of the input data.
            output_size (int): The size of the output data.
            lr (float): The learning rate for training.
        """
        super(DeepSELEX, self).__init__(output_size)
        self.save_hyperparameters()
        self.seq_size = seq_size
        self.k = k
        self.output_size = output_size
        self.sub_seq = seq_size - k + 1
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=512, kernel_size=8, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)

        # the input size is 4*k, after conv1 the size is 512*(k-7) then after maxpooling the size is 512*(k-7)//5
        self.fc1 = nn.Linear(512*((k-7)//5), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.lr = lr
        
    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output data.
        """
        # Transpose the input: from (batch, sub-seq, seq_length, channels) to (batch, sub-seq, channels, seq_length)
        x = x.permute(0, 1, 3, 2)

        # Track the original batch size and sub-seq dimensions
        batch_size = x.size(0)
        sub_seq = x.size(1)

        # Then reshape to (batch_size * sub_seq, channels, seq_length)
        x = x.reshape(-1, x.size(2), x.size(3))

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        # reshape to (batch_size, sub_seq, output_size)
        x = x.reshape(batch_size, sub_seq, self.output_size)
        return x

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
        # Forward pass to get outputs from the model
        outputs = self(X)  # Shape: (batch_size, sub_seq, output_size)

        # y shape: batch_size so we need to expand it to (batch_size, sub_seq)
        y = y.unsqueeze(1).expand(-1, self.sub_seq)

        # Calculate loss using raw logits (not argmax)
        loss = F.cross_entropy(outputs.reshape(-1, self.output_size), y.reshape(-1))
        # Predictions: take the argmax after calculating loss
        self.log(f'train_loss', loss, prog_bar=True, logger=True)
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
        # Forward pass to get outputs from the model
        outputs = self(X)  # Shape: (batch_size, sub_seq, output_size)

        # y shape: batch_size so we need to expand it to (batch_size, sub_seq)
        y = y.unsqueeze(1).expand(-1, self.sub_seq)

        # Calculate loss using raw logits (not argmax)
        loss = F.cross_entropy(outputs.reshape(-1, self.output_size), y.reshape(-1))
        self.log(f'val_loss', loss, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for the model.
        Args:
            batch (tuple): A tuple containing the input data.
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The predicted values for the batch.
        """
        # X shape: (batch_size, sub_seq, seq_length, channels)
        X = batch[0]

        # Get predictions from the model
        prediction = self(X)  # Prediction shape: (batch_size, sub_seq, output_size)

        if self.output_size == 2:  # in case output_size <= 2
            max_last = torch.max(prediction[:, :, -1], dim=1)[0]  # Max over sub_seq for last output
            min_first_0 = torch.min(prediction[:, :, 0], dim=1)[0]  # Min over sub_seq for the first output
            return max_last - min_first_0
        max_last_4 = torch.max(prediction[:, :, -1], dim=1)[0]  # Max over sub_seq for last output
        max_last_3 = torch.max(prediction[:, :, -2], dim=1)[0]  # Max over sub_seq for second last output
        min_first_0 = torch.min(prediction[:, :, 0], dim=1)[0]  # Min over sub_seq for the first output

        # Compute the final result for each batch item
        res = max_last_4 + max_last_3 - min_first_0

        return res