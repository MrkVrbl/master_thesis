import numpy as np
import pandas as pd
import random
#from src.models.baseline import *

def get_gc_count_from_ohe(train_df):
    """
    Calculates GC content in sequence for positive and negative sequences separately
    """
    X_train, y_train = get_X_y(train_df)

    pos_counts = X_train[y_train == 1].reshape(-1, 4).sum(axis=0)
    neg_counts = X_train[y_train == 0].reshape(-1, 4).sum(axis=0)

    pos_gc = (sum(pos_counts[1:3]) * 100) / sum(pos_counts)
    neg_gc = (sum(neg_counts[1:3]) * 100) / sum(neg_counts)

    return pos_gc, neg_gc
    

def get_gc_count_from_seq(seqs):
    gc = 0
    count = 0

    for seq in seqs:
        for char in seq:
            count += 1
            if char == "g" or char == "c":
                gc += 1

    return gc * 100 / count


def seq_to_ohe(row):
    """
    Turn nucleotide sequence to one hot encoding
    """
    seq = row["seq"].upper()
    nucleotid = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1], '0':[0,0,0,0], 'N':[0.25,0.25,0.25,0.25]}
    ohe_seq = np.array([nucleotid[x] for x in seq])

    return np.concatenate([ohe_seq], axis=1)


def get_X_y(df):
    """
    Devide df to X,y, apply changes to X, y is usually label
    """
    return np.stack(df.apply(seq_to_ohe, axis=1)), df.label

"""
def concat_df(pos, neg):

    #Concantenates two dfs

    return pd.concat([pos, neg], ignore_index=True).sample(frac=1).reset_index(drop=True)


def shuffle(df):

    #for each sequence in dataset, shuffle nucleotides in given sequence

    return df['seq'].map(lambda seq: ''.join(random.sample(seq, k=len(seq))))


def shuffle_both(pos_train, neg_train):

    #concatenate pos_train and neg_train to form train df, than shuffle sequences in whole df

    train_df = concat_df(pos_train, neg_train)
    train_df['seq'] = shuffle(train_df)

    return train_df


def neg_as_shuflled_pos(pos_train, neg_train):

    #shuffle pos_train and asign it to neg_train, concatanted both to train_df

    pos_train_unshufled = pos_train.copy(deep=True)
    neg_train['seq'] = shuffle(pos_train)
    neg_train.dropna(subset=["seq"], inplace=True)
    train_df = concat_df(pos_train_unshufled, neg_train)

    return train_df


def augment_negatives(dfs):

    #devided dataset (pos_train, neg_train, pos_ls, neg_ls) concatenate to train_df, test_df
    #orig_train_df = concatenated without changes
    #shuffled_train_df = both pos_train and neg_train are shuffled (only information left is difference in GC content)
    #sameGC_train_df = neg_train as shuffled pos_train (no information)
    #test_df = concatented pos_ls, neg_ls (untouched)

    print("    Augmenting negatives...")

    pos_train, neg_train, pos_ls, neg_ls = dfs

    orig_train_df = concat_df(pos_train, neg_train)
    shuffled_trian_df = shuffle_both(pos_train, neg_train)
    sameGC_train_df = neg_as_shuflled_pos(pos_train, neg_train)
    #add more 
    test_df = concat_df(pos_ls, neg_ls)

    return {"original":[orig_train_df, test_df], 
            "shuffled":[shuffled_trian_df, test_df],
            "sameGC":[sameGC_train_df, test_df]}


def train_baseline(datasets, content, protein_name, pos_gc, neg_gc):

    #For each created dataset, transforms sequences to OHE, runs baseline model (LogReg), writes results to content.

    print(f"    Training baseline models...")

    for df_type, datasets in datasets.items():
        train_df = datasets[0]
        test_df = datasets[1]
        X_train, y_train = get_X_y(train_df)
        X_test, y_test = get_X_y(test_df)
        train_auc_score, test_auc_score, baseline = baseline_model(X_train, X_test, y_train, y_test, max_iter=200)
        content.append([protein_name, pos_gc, neg_gc, train_auc_score, test_auc_score, df_type])

    return content

"""