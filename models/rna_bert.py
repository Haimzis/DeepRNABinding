import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC
from torchmetrics.classification import MulticlassAccuracy
from multimolecule import RnaBertModel, RnaTokenizer


class RnaBert(pl.LightningModule):
    def __init__(self, output_size, lr=0.003):
        super(RnaBert, self).__init__()
        self.accuracy = MulticlassAccuracy(num_classes=output_size)
        self.auroc = AUROC(num_classes=output_size, task="multiclass")

        self.lr = lr

        # Load pre-trained RNABERT model
        self.rna_bert = RnaBertModel.from_pretrained('multimolecule/rnabert')

        # Classification head
        self.fc1 = nn.Linear(self.rna_bert.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Optionally freeze RNABERT if not fine-tuning
            outputs = self.rna_bert(input_ids, attention_mask=attention_mask).last_hidden_state

        # Use the [CLS] token representation for classification
        cls_output = outputs[:, 0, :]

        # Pass through classification head
        x = F.relu(self.fc1(cls_output))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, _, y = batch
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, _, y = batch
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, y)
        return loss

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch[0], batch[1]
        outputs = self(input_ids, attention_mask)
        return torch.argmax(outputs, dim=1)
