import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from data_processing.utils import compute_ss_df

in_dir_path = Path("/home/mrkvrbl/Diplomka/Data/prisment/raw_tsv")
out_dir_path = Path("/home/mrkvrbl/Diplomka/Data/prisment/processed")


def save_dfs(train_positives, train_negatives, test_positives, test_negatives, dir_path, rbp):
    #save to coresponding directories
    path = Path(str(dir_path) + "/" + "positives_train")
    path.mkdir(exist_ok=True)
    train_positives.to_csv(str(path) + "/" + rbp, sep="\t", header=False, index=False)

    path = Path(str(dir_path) + "/" + "positives_ls")
    path.mkdir(exist_ok=True)
    test_positives.to_csv(str(path) + "/" + rbp, sep="\t", header=False, index=False)

    path = Path(str(dir_path) + "/" + "negatives_train")
    path.mkdir(exist_ok=True)
    train_negatives.to_csv(str(path) + "/" + rbp, sep="\t", header=False, index=False)

    path = Path(str(dir_path) + "/" + "negatives_ls")
    path.mkdir(exist_ok=True)
    test_negatives.to_csv(str(path) + "/" + rbp, sep="\t", header=False, index=False)


def prism_preprocessing(in_dir_path=in_dir_path, out_dir_path=out_dir_path,compute_secondary_structure=True):

    out_dir_path.mkdir(exist_ok=True)
    rbps = os.listdir(in_dir_path)
    count = 0

    print(f'Number of files to process: {len(rbps)}')

    for rbp in rbps:
        if rbp.endswith('.tsv.gz'):
            print(f"Processing file: {rbp}")

            count += 1
            rbp_path = str(in_dir_path) + "/" + rbp

            df = pd.read_csv(rbp_path, delimiter = "\t", header=1, names=['type', 'name', 'seq', 'icshape', 'score', 'label'], compression='gzip')
            df.drop(columns=['type', 'icshape', 'score'], inplace=True, )

            if compute_secondary_structure:
                compute_ss_df(df)

            train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

            train_positives = train_df[train_df['label']==1]
            train_negatives = train_df[train_df['label']==0]
            test_positives = test_df[test_df['label']==1]
            test_negatives = test_df[test_df['label']==0]
            
            save_dfs(train_positives, train_negatives, test_positives, test_negatives, out_dir_path, rbp[:-3])

            print(f"File number {count} processed")
            print("...")