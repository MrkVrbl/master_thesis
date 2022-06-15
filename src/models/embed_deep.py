import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from models.multihead_attention import MultiheadAttention


class ECLAD(pl.LightningModule):

    def __init__(self, config):
        super(ECLAD, self).__init__() 

        self.save_hyperparameters(config)

        self.learning_rate = self.hparams.learning_rate
        self.decay_factor = self.hparams.decay_factor
        self.batch_size = self.hparams.batch_size
        
        self.auroc = AUROC(num_classes=1)
        self.acc = Accuracy()

        # Emmbedding
        self.embedd = nn.Embedding(self.hparams.vocab_size, self.hparams.embedding_dim, padding_idx=0)

        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.hparams.embedding_dim, out_channels=self.hparams.CONV1_out_channels, kernel_size=self.hparams.CONV1_kernelsize, padding="same"),
            nn.ReLU(), 
            nn.Dropout(0.5))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hparams.CONV1_out_channels, out_channels=self.hparams.CONV2_out_channels, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(8))
        
        #self.conv3 = nn.Conv1d(in_channels=self.hparams.num_channels, out_channels=self.hparams.num_channels // 2, kernel_size=1, padding=0, bias=True)
        
        # Biderectional LSTM
        self.lstm = torch.nn.LSTM(
            input_size=self.hparams.CONV1_out_channels,
            hidden_size=self.hparams.LSTM_num_features,
            num_layers=1,
            #dropout=0.25,
            bidirectional=True,
            batch_first=True)

        # Multihead attention
        self.multihead_attn = MultiheadAttention(input_dim=self.hparams.LSTM_num_features*2, embed_dim=self.hparams.LSTM_num_features*2, num_heads=4) 

        self.flatten = nn.Flatten()

        if self.hparams.DIMRED:
            self.hparams.num_channels = self.hparams.num_channels // 2
      
        if self.hparams.LSTM:
            self.linear = nn.Sequential(
                nn.Linear((self.hparams.seq_lenght*self.hparams.LSTM_num_features*2)//8, self.hparams.DENSE_kernelsize), #self.hparams.seq_lenght*self.hparams.LSTM_num_features*2
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(self.hparams.DENSE_kernelsize),
                nn.Linear(self.hparams.DENSE_kernelsize, 2))
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.hparams.embedding_dim*2*128, self.hparams.DENSE_kernelsize),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.BatchNorm1d(self.hparams.DENSE_kernelsize),
                nn.Linear(self.hparams.DENSE_kernelsize, 2))


    def forward(self, inputs):

        #print(f"Input shape: {inputs.shape}")
        embeds = self.embedd(inputs).permute(0,2,1) #batch, embed_dim, seq_lenght
        #print(f"Embedding shape: {embeds.shape}")
        x = self.conv1(embeds) #batch, embed_dim, seq_lenght
        #print(f"Conv1 output: {x.shape}")
        if self.hparams.CONV2:
            x = self.conv2(x) #batch, embed_dim, seq_lenght
            #print(f"Conv2 output: {x.shape}")
        if self.hparams.LSTM:
            x = x.permute(0,2,1) #batch, seq_lenght, embed_dim
            #print(f"LSTM input: {x.shape}")
            x,_ = self.lstm(x) #batch, seq_lenght , embed_dim
            #print(f"LSTM output: {x.shape}")
        if self.hparams.ATTN:
            x = self.multihead_attn(x) #batch, seq_lenght , embed_dim
            #print(f"ATTN output: {x.shape}")
            x = x.permute(0,2,1) #batch, embed_dim, seq_lenght
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

        return test_auroc_epoch

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch['seq'], batch['label']
        outputs = self(inputs)
        labels = labels.cpu().detach()
        preds = torch.max(outputs, 1)[1].cpu().detach()

        return inputs, labels, preds