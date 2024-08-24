import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy


class BaseModel(pl.LightningModule):
    def __init__(self, output_size):
        """
        Base class for models to handle metrics initialization.
        
        Args:
            output_size (int): The number of classes or output features.
        """
        super(BaseModel, self).__init__()

        # Initialize common metrics
        self.accuracy = MulticlassAccuracy(num_classes=output_size)

    def log_metrics(self, loss, preds, targets, stage):
        """
        Log the metrics for a given stage (train/val/test).
        
        Args:
            loss (torch.Tensor): The loss value.
            preds (torch.Tensor): The predicted values.
            targets (torch.Tensor): The ground truth labels.
            stage (str): The stage of logging (train, val, test).
        """
        acc = self.accuracy(preds, targets)
        self.log(f'{stage}_loss', loss, prog_bar=True, logger=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, logger=True)