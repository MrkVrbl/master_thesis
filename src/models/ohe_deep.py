import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from models.multihead_attention import MultiheadAttention


class OCLAD(pl.LightningModule):

    def __init__(self, config):
        super(OCLAD, self).__init__() 

        self.save_hyperparameters(config)

        self.learning_rate = self.hparams.learning_rate
        self.decay_factor = self.hparams.decay_factor
        self.batch_size = self.hparams.batch_size
        
        self.auroc = AUROC(num_classes=1)
        self.acc = Accuracy()

        # conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=self.hparams.CONV1_kernelsize, padding="same"),
            nn.ReLU(), 
            nn.Dropout(0.25))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=self.hparams.num_channels, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(8))
        
        self.lstm = torch.nn.LSTM(
            input_size=128,
            hidden_size=self.hparams.LSTM_kernelsize,
            num_layers=1,
            dropout=0.25,
            bidirectional=True,
            batch_first=True)

        self.multihead_attn = MultiheadAttention(input_dim=self.hparams.num_channels, embed_dim=self.hparams.num_channels, num_heads=4)

        self.conv3 = nn.Conv1d(in_channels=self.hparams.num_channels, out_channels=self.hparams.num_channels // 2, kernel_size=1, padding=0, bias=True)

        self.flatten = nn.Flatten()

        if self.hparams.DIMRED:
            self.hparams.num_channels = self.hparams.num_channels // 2
      
        if self.hparams.LSTM:
            self.linear = nn.Sequential(
                nn.Linear(self.hparams.num_channels*self.hparams.LSTM_kernelsize*2, self.hparams.DENSE_kernelsize),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.BatchNorm1d(self.hparams.DENSE_kernelsize),
                nn.Linear(self.hparams.DENSE_kernelsize, 2))
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.hparams.num_channels*128, self.hparams.DENSE_kernelsize),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.BatchNorm1d(self.hparams.DENSE_kernelsize),
                nn.Linear(self.hparams.DENSE_kernelsize, 2))


    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)
        if self.hparams.CONV2:
            x = self.conv2(x)
            #print(x.shape)
        if self.hparams.LSTM:
            x,_ = self.lstm(x)
            #print(f"LSTM output: {x.shape}")
        if self.hparams.ATTN:
            x = x.permute(0,2,1)
            #print(f"permute output: {x.shape}")
            x = self.multihead_attn(x)
            #print(f"ATTN output: {x.shape}")
            x = x.permute(0,2,1)
            #print(f"permute output: {x.shape}")
        if self.hparams.DIMRED:
            x = self.conv3(x)
            #print(f"DimReduction output: {x.shape}")
        x = self.flatten(x)
        #print(f"Flatten output: {x.shape}")
        x = self.linear(x)
        #print(x.shape)
  
        return F.log_softmax(x, dim=-1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor ** epoch)
        return [optimizer], [scheduler] 

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['seq'], batch['label']
        outputs = self(inputs)
        preds = torch.max(outputs, 1)[1]

        criterion = nn.NLLLoss()
        loss = criterion(outputs, labels)

        labels = labels.cpu().detach().int()
        preds = preds.cpu().detach().int()

        train_acc = self.acc(preds, labels)
        train_auroc = self.auroc(preds, labels)
        
        return {"loss": loss,
                "train_acc": train_acc,
                "train_auroc": train_auroc}
                

    def training_epoch_end(self, train_step_outputs):
        loss = torch.stack([x["loss"] for x in train_step_outputs]).mean()
        train_acc_epoch = torch.stack([x["train_acc"] for x in train_step_outputs]).mean()
        train_auroc_epoch = torch.stack([x["train_auroc"] for x in train_step_outputs]).mean()
        
        self.log("train/epoch/loss", loss)
        self.log("train/epoch/acc", train_acc_epoch)
        self.log("train/epoch/auroc", train_auroc_epoch) 
    
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['seq'], batch['label']
        outputs = self(inputs)

        criterion = nn.NLLLoss()
        loss = criterion(outputs, labels)

        labels = labels.cpu().detach()
        preds = torch.max(outputs, 1)[1].cpu().detach()

        val_acc = self.acc(preds, labels)
        val_auroc = self.auroc(preds, labels)
        
        return {"loss": loss,
                "val_acc": val_acc,
                "val_auroc": val_auroc}


    def validation_epoch_end(self, val_step_outputs):
        loss = torch.stack([x["loss"] for x in val_step_outputs]).mean()
        val_acc_epoch = torch.stack([x["val_acc"] for x in val_step_outputs]).mean()
        val_auroc_epoch = torch.stack([x["val_auroc"] for x in val_step_outputs]).mean()
        
        self.log("val/epoch/loss", loss)
        self.log("val/epoch/acc", val_acc_epoch)
        self.log("val/epoch/auroc", val_auroc_epoch)

    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch['seq'], batch['label']
        outputs = self(inputs)

        criterion = nn.NLLLoss()
        loss = criterion(outputs, labels)

        labels = labels.cpu().detach()
        preds = torch.max(outputs, 1)[1].cpu().detach()

        test_acc = self.acc(preds, labels)
        test_auroc = self.auroc(preds, labels)
        
        return {"loss": loss,
                "test_acc": test_acc,
                "test_auroc": test_auroc}

    def test_epoch_end(self, test_step_outputs):
        loss = torch.stack([x["loss"] for x in test_step_outputs]).mean()
        test_acc_epoch = torch.stack([x["test_acc"] for x in test_step_outputs]).mean()
        test_auroc_epoch = torch.stack([x["test_auroc"] for x in test_step_outputs]).mean()
        
        self.log("test/loss", loss)
        self.log("test/acc", test_acc_epoch)
        self.log("test/auroc", test_auroc_epoch)


    def predict_step(self, batch, batch_idx):
        inputs, labels = batch['seq'], batch['label']
        outputs = self(inputs)
        labels = labels.cpu().detach()
        preds = torch.max(outputs, 1)[1].cpu().detach()

        return inputs, labels, preds