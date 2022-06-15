import re
import os
from pathlib import Path
import pandas as pd
import numpy as np
import site
import RNA
import pandas as pd
import pybedtools
site.addsitedir('/usr/local/lib/python3/site-packages/')  # Always appends to end


def adjust_window(line, window_size):
    #Adjust coordinate window of a bed file
    start = line.start
    end = line.stop
    length = end - start
    middle = start + (length / 2)
    
    new_start = middle - (window_size/2)
    new_end = middle + (window_size/2)
    
    line.start = new_start
    line.stop = new_end
    return line


def remove_redundant(root, negatives_path, positives_path):
    os.remove(negatives_path)
    os.remove(positives_path)


def concat_to_tsv(protein_path):
    for root, _, files in os.walk(protein_path):
        if files:
            negatives_path = root + "/negatives.tsv"
            positives_path = root + "/positives.tsv"
            out_file = root + "/original.tsv.gz"

            negatives_df = pd.read_csv(negatives_path, sep='\t', names=['id', 'seq'])
            negatives_df['label'] = 0

            positives_df = pd.read_csv(positives_path, sep='\t', names=['id', 'seq'])
            positives_df['label'] = 1

            df = pd.concat([negatives_df, positives_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
            df.to_csv(out_file, sep='\t', compression='gzip')

            remove_redundant(root, negatives_path, positives_path)


def map_cons(bed_file, conservation_arrays_path):

    print(f"    Mapping conservation scores...")

    dict_of_cons_arrays = {re.compile(r'chr[0-9A-Z]*').findall(str(file.stem))[0] : str(file) for file in conservation_arrays_path.glob('*.npy')}

    dict_of_intervals_per_chrom = {}
    for chromosome in dict_of_cons_arrays.keys():
        per_chrom_intervals = []
        with open(bed_file, "r") as bed:
            for line in bed:
                chrom, interval = str(str(line.split()[0]).split(":")[0]), str(str(line.split()[0]).split(":")[1])[:-2]
                if chromosome == chrom:
                    per_chrom_intervals.append(f'{interval}') 
            dict_of_intervals_per_chrom[chromosome] = per_chrom_intervals

    scores = []
    for k,v in dict_of_cons_arrays.items():
        cons_array = np.load(v)
        same_chrom_lines = dict_of_intervals_per_chrom[k]
        for interval in same_chrom_lines:
            arr_slice = cons_array[int(interval.split('-')[0]):int(interval.split('-')[1])]
            list_of_scores = [float(i) for i in arr_slice]
            scores.append(','.join(str(score) for score in list_of_scores))

    content = ""
    with open(bed_file, "r") as bed_in:
        for i, line in enumerate(bed_in):
            fields = line.strip().split('\t')
            if i < len(scores):
                fields.append(str(scores[i]))
            content = content + '\t'.join(fields) + '\n'
    with open(bed_file, 'w') as bed_out:
        bed_out.write(content)


def compute_ss(bed_file):
    #Add secondary structure notation to the end of each line of a file
    print(f"    Computing secondary structure...")
    content = ""
    with open(bed_file, 'r') as file:
        for line in file:
            fields = line.strip().split('\t')
            sec_structure, _ = RNA.fold(fields[1])
            fields.append(sec_structure)
            content = content + '\t'.join(fields) + '\n'
    with open(bed_file, 'w') as fout:
        fout.write(content)


def compute_ss_df(df):
    #Add secondary structure notation to the dataframe
    print(f"    Computing secondary structure...")
    df["sec_structure"] = df.seq.apply(lambda x: RNA.fold(x)[0])

def intersection(a, b):
    a = pybedtools.BedTool(a)
    b = pybedtools.BedTool(b)

    return a.intersect(b, u=True)


def create_df(file, label):
    """
    Creates pandas df, when creating pandas df for rbp protein, adds label
    """
    df = pd.read_csv(file, delimiter = "\t")

    if len(df.iloc[0]) > 2:
        df.columns = ['name', 'seq', 'label']

    else:
        df.columns = ['name', 'seq']
        labels = [label for i in range(len(df))]
        df['label'] = labels

    return df


def csv_to_df(dir_dict, i):
    """
    Iterated over subdirs with same index(i), for each subdir creates coresponding dataset
    """
    for subdir in dir_dict.keys():
        fields = subdir.split("/")
        subdir_name = fields[-1]
       
        if subdir_name == "positives_train":
            file = dir_dict[subdir][i]
            path_to_file = subdir + "/" + file
            pos_train = create_df(path_to_file, 1)

        elif subdir_name == "negatives_train":
            file = dir_dict[subdir][i]
            path_to_file = subdir + "/" + file
            neg_train = create_df(path_to_file, 0)

        elif subdir_name == "positives_ls":
            file = dir_dict[subdir][i]
            path_to_file = subdir + "/" + file
            pos_ls = create_df(path_to_file, 1)

        elif subdir_name == "negatives_ls":
            file = dir_dict[subdir][i]
            path_to_file = subdir + "/" + file
            neg_ls = create_df(path_to_file, 0)

    return (pos_train, neg_train, pos_ls, neg_ls)


def dir_listing(dir_path):
    """
    subdirs_path = full path to each subdir (positve_ls, positives_train...)
    dir_dict = dictionary{subdir: list of file in subdir}
    entry_count = number of entries(files) in given subdir
    """
    subdirs_path = [str(str(dir_path) + "/" + subdir) for subdir in os.listdir(dir_path)]
    dir_dict = {subdir : sorted([file for file in os.listdir(subdir)]) for subdir in subdirs_path} 
    entry_count = len(list(dir_dict.values())[0])

    return dir_dict, entry_count


def process_dataset(rbp_dir_path, prism_dir_path, dataset='rbp24'):

    """
    Processing each data folder (rbp, prism), data in folders are distributed to four subdris
    (pos_train, neg_train, pos_ls, neg_ls) each contains coresponding .tsv files

    Goes through each file in each subdir
    """

    if dataset == "rbp24":
        rbp_dir_dict, rbp_entry_count = dir_listing(rbp_dir_path)
        return rbp_dir_dict, rbp_entry_count

    elif dataset == "prism":
        prism_dir_dict, prism_entry_count = dir_listing(prism_dir_path)
        return prism_dir_dict, prism_entry_count

    else:
        print("Wrong dataset specification! Choose between rbp24, prism")
        return
