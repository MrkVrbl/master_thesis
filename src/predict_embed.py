import json
import os
import pandas as pd
from tokenizers import Tokenizer
from pytorch_lightning import Trainer
from torchmetrics import AUROC

from models.embed_deep import ECLAD
from data_loader.embed_dataloader import make_dataloaders
from utils.utils import get_gc_count_from_seq


best_models_path = "/home/mrkvrbl/Diplomka/best_models/"
proteins = sorted(os.listdir(best_models_path))
results_path = "/home/mrkvrbl/Diplomka/Data/outputs/rbp24_deep_embed_results.tsv"

with open(results_path, "w") as out_f:
    out_f.write(f"Name\tAUC\tFP\tFP_GC\tFN\tFN_GC\n")


for protein in proteins:

    protein_path = best_models_path + "/" + protein
    params_path = protein_path + "/params.json"
    checkpoint_path = protein_path + "/best.ckpt" 

    with open(params_path) as f_in:
        PARAMS = json.loads(f_in.read())

    tokenizer = Tokenizer.from_file(f'/home/mrkvrbl/Diplomka/Data/tokenizers/transcriptome_hg19_{PARAMS["vocab_size"]}words_bpe.tokenizer.json')
    test_model = ECLAD(PARAMS)
    test_model.load_from_checkpoint(checkpoint_path)


    train_df = pd.read_csv(PARAMS["train_df_path"], delimiter="\t", names=["seq", "label"])
    test_df = pd.read_csv(PARAMS["test_df_path"], delimiter="\t", names=["seq", "label"])

    longest_seq = max(max([len(tokenizer.encode(seq).ids) for seq in train_df.seq]), max([len(tokenizer.encode(seq).ids) for seq in test_df.seq]))
    PARAMS['seq_lenght'] = longest_seq

    _, _, testloader = make_dataloaders(train_df, test_df, tokenizer, longest_seq, PARAMS["batch_size"], PARAMS["num_workers"], PARAMS["val_split"])

    test_trainer = Trainer()
    predict_out = test_trainer.predict(test_model, ckpt_path=checkpoint_path, dataloaders=testloader, return_predictions=True)

    inputs = []
    labels = []
    preds = []
    for outs in predict_out:
        inputs.append(outs[0])
        labels.append(outs[1])
        preds.append(outs[2])

    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    preds = [item for sublist in preds for item in sublist]
    FP, FN = [], []

    count = 0
    for i in range(len(inputs)):
        if labels[i] < preds[i]:
            FP.append(inputs[i])
        elif labels[i] > preds[i]:
            FN.append(inputs[i])
        count += 1

    FP = ["".join(tokenizer.decode(FP[i].numpy()).split(" ")) for i in range(len(FP))]
    FN = ["".join(tokenizer.decode(FN[i].numpy()).split(" ")) for i in range(len(FN))]

    fp_gc = get_gc_count_from_seq(FP)
    fn_gc = get_gc_count_from_seq(FN)

    with open(results_path, "a") as out_f:
        out_f.write(f"{PARAMS['name']}\t{PARAMS['best_auc']}\t{len(FP)}\t{fp_gc}\t{len(FN)}\t{fn_gc}\n")

