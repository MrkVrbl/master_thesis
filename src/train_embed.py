import pandas as pd
from pathlib import Path
import shutil
import json
import os

from tokenizers import Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers.neptune import NeptuneLogger
from torchmetrics import AUROC

from data_processing.utils import *
from models.embed_deep import ECLAD
from data_loader.embed_dataloader import make_dataloaders
from utils.utils import get_gc_count_from_seq



PARAMS = {
        "model": "ECLAD",
        "vocab_size": 16,
        "embedding_dim": 8,
        "CONV1_kernelsize": 9,
        "CONV1_out_channels": 12,
        "CONV2": False,
        "CONV2_out_channels": 16,
        "LSTM": True,
        "LSTM_num_features": 8,
        "ATTN": True,
        "DIMRED": False,
        "DENSE_kernelsize":64,
        "batch_size": 256,
        "learning_rate": 0.005,
        "decay_factor": 0.95,
        "max_epochs": 100,
        "num_workers": 16,
        "val_split": 0.1
    }



dataset_path = "/home/mrkvrbl/Diplomka/Data/rbp24/extracted_seqs"
tokenizer = Tokenizer.from_file(f'/home/mrkvrbl/Diplomka/Data/tokenizers/transcriptome_hg19_{PARAMS["vocab_size"]}words_bpe.tokenizer.json')
proteins = sorted(os.listdir(dataset_path))

for protein in proteins:

    PARAMS['name'] = protein

    #define train, test path
    train_df_path = Path(dataset_path + "/" + protein +  "/train/original.tsv")
    PARAMS["train_df_path"] = str(train_df_path)
    test_df_path = Path(dataset_path + "/" + protein +  "/test/original.tsv")
    PARAMS["test_df_path"] = str(test_df_path)

    #create train, test dataframes
    train_df = pd.read_csv(train_df_path, delimiter="\t", names=["seq", "label"])
    test_df = pd.read_csv(test_df_path, delimiter="\t", names=["seq", "label"])

    #specify longest sequence (necessary for padding)
    longest_seq = max(max([len(tokenizer.encode(seq).ids) for seq in train_df.seq]), max([len(tokenizer.encode(seq).ids) for seq in test_df.seq]))
    PARAMS['seq_lenght'] = longest_seq

    #create dataloaders
    trainloader, valloader, testloader = make_dataloaders(train_df, test_df, tokenizer, longest_seq, PARAMS["batch_size"], PARAMS["num_workers"], PARAMS["val_split"])

    #define early stopping
    early_stopping = EarlyStopping('val/epoch/loss', patience=10, check_on_train_epoch_end=False, )

    #prepare for saving model checkpoints
    checkpoints_path = "checkpoints/" + PARAMS["name"]
    model_checkpoint = ModelCheckpoint(
            dirpath=checkpoints_path,
            filename="{epoch:02d}",
            save_weights_only=True,
            save_top_k=-1,
            save_last=True,
            monitor="val/epoch/loss",
            every_n_epochs=1)

    # create NeptuneLogger
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZDI3YzE1Yy0yYzllLTRjM2YtYjk2MS1jNzNiZmI3MzIyNWEifQ==",  # replace with your own
        project="mrkvrbl/MasterThesis",  # "<WORKSPACE/PROJECT>"
        name=PARAMS["name"])    


    #define trainer
    trainer = Trainer(logger=neptune_logger,
                    callbacks=[model_checkpoint, early_stopping],
                    max_epochs=PARAMS['max_epochs'],
                    accumulate_grad_batches=1,
                    gradient_clip_val=0.5,
                    stochastic_weight_avg=True,
                    gpus=1)

    #define model
    model = ECLAD(PARAMS)

    #log model_summary and hyperparemeters
    neptune_logger.log_model_summary(model=model, max_depth=-1)
    neptune_logger.log_hyperparams(params=PARAMS)

    # TRAIN 
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

    # TEST
    checkpoints = sorted(os.listdir(checkpoints_path))
    best_auc = 0

    for checkpoint in checkpoints:
        checkpoint_path = str(checkpoints_path) + "/" + checkpoint
        test_out = trainer.test(model, ckpt_path=checkpoint_path, dataloaders=testloader)[0]

        if test_out["test/auroc"] > best_auc:
            best_auc = test_out["test/auroc"]
            best_checkpoint_path = checkpoint_path


    # save best model   
    protein_best_model_path = Path("/home/mrkvrbl/Diplomka/best_models/" + PARAMS['name'])
    PARAMS["best_auc"] = best_auc

    if protein_best_model_path.exists():
        shutil.rmtree(protein_best_model_path)
    protein_best_model_path.mkdir()

    protein_best_ckp_path = str(protein_best_model_path) +"/best.ckpt"
    shutil.copyfile(best_checkpoint_path, protein_best_ckp_path)

    # save PARAMS
    json_path = str(protein_best_model_path) +"/params.json"
    with open(json_path, 'w') as outfile:
        json.dump(PARAMS, outfile)

    shutil.rmtree(checkpoints_path)
