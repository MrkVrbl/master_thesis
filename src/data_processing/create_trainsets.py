from dataclasses import replace
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path


rbp31_path = "/home/mrkvrbl/Diplomka/Data/rbp31"
rbp24_path = "/home/mrkvrbl/Diplomka/Data/rbp24/processed"


def concat_df(positives, negatives):
    """
    Concantenates two dfs
    """
    return pd.concat([positives, negatives], ignore_index=True).sample(frac=1).reset_index(drop=True)


def shuffle(df, n_samples, replace=False):
    """
    for each sequence in dataset, shuffle nucleotides in given sequence
    """
    return df['seq'].map(lambda seq: ''.join(random.sample(seq, k=len(seq)))).sample(n=n_samples, replace=replace)


def shuffle_both(positives, negatives):
    """
    concatenate positives and negatives to form train df, than shuffle sequences in whole df
    """
    df = concat_df(positives, negatives)
    df['seq'] = shuffle(df, len(df))
    return df


def neg_as_shuflled_pos(positives, negatives):
    """
    shuffle positives and asign it to negatives, concatanted both to df
    """
    positives_unshufled = positives.copy(deep=True)
    negatives.loc[:,'seq'] = list(shuffle(positives, len(negatives), replace=True))
    df = concat_df(positives_unshufled, negatives)
    return df


for root, dirs, files in os.walk(rbp24_path):
    if root.endswith('train'):
        original = pd.read_csv(root + '/original.tsv.gz', delimiter='\t', index_col=0, header=0, compression="gzip")
        print(original)
                
        positives = original[original.label == 1]
        negatives = original[original.label == 0]

        shuffled = shuffle_both(positives, negatives)
        sameGC = neg_as_shuflled_pos(positives, negatives)

        shuffled.to_csv(root + '/shuffled.tsv.gz', sep='\t', compression="gzip")
        sameGC.to_csv(root + '/sameGC.tsv.gz', sep='\t', compression="gzip")

        print(root)