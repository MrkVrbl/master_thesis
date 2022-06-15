import numpy as np
import pandas as pd
import random
import os
import pybedtools
from pathlib import Path
from utils import intersection

rbp31_path = "/home/mrkvrbl/Diplomka/Data/rbp31"
rbp24_path = "/home/mrkvrbl/Diplomka/Data/rbp24"

random_transcripts_path = "/home/mrkvrbl/Diplomka/Data/outputs/random_transcripts.csv"

introns_bed = "/home/mrkvrbl/Diplomka/Data/reference/region_types/introns.bed.gz"
exons_bed = "/home/mrkvrbl/Diplomka/Data/reference/region_types/exons.bed.gz"
utr5_bed = "/home/mrkvrbl/Diplomka/Data/reference/region_types/5utrs.bed.gz"
utr3_bed = "/home/mrkvrbl/Diplomka/Data/reference/region_types/3utrs.bed.gz"

introns_tsv = "/home/mrkvrbl/Diplomka/Data/reference/region_types/introns_intersections.tsv"
exons_tsv = "/home/mrkvrbl/Diplomka/Data/reference/region_types/exons_intersections.tsv"
utr5_tsv = "/home/mrkvrbl/Diplomka/Data/reference/region_types/utr5_intersections.tsv"
utr3_tsv = "/home/mrkvrbl/Diplomka/Data/reference/region_types/utr3_intersections.tsv"

regions_bed = [introns_bed, exons_bed, utr5_bed, utr3_bed]
regions_tsv = [introns_tsv, exons_tsv, utr5_tsv, utr3_tsv]



def generate_distributed_sequences(sequences_len, sequences_bed, regions_bed, regions_tsv):
    region_counts = []
    sequences = []

    for region in regions_bed:
        intersect = intersection(sequences_bed, region)
        region_counts.append(len(intersect))

    regions_total_count = sum(region_counts)
    region_ratios = [count / regions_total_count for count in region_counts]
    region_final_counts = [int(sequences_len * ratio) for ratio in region_ratios]
    
    for i in range(4):
        df = pd.read_csv(regions_tsv[i], delimiter="\t", names=['id', 'seq']).sample(n=region_final_counts[i])
        sequences.append(df)

    return pd.concat(sequences, ignore_index=True).sample(frac=1)

def pick_distributed_positives(negatives_df, positives_len, positives_bed, regions_bed, regions_tsv):
    positives_df = generate_distributed_sequences(positives_len, positives_bed, regions_bed, regions_tsv)
    positives_df['label'] = 1
    random_pos_df = pd.concat([negatives_df, positives_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

    return random_pos_df

def pick_distributed_negatives(positives_df, negatives_len, positives_bed, regions_bed, regions_tsv):
    negatives_df = generate_distributed_sequences(negatives_len, positives_bed, regions_bed, regions_tsv)
    negatives_df['label'] = 0
    distributed_neg_df = pd.concat([positives_df, negatives_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

    return distributed_neg_df

def pick_random_positives(negatives_df, positives_len, transcripts_path):
    positives_df = pd.read_csv(transcripts_path, delimiter="\t", names=["id", "seq"]).sample(n=positives_len)
    positives_df['label'] = 1
    random_df = pd.concat([positives_df, negatives_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

    return random_df


for root, dirs, files in os.walk(rbp31_path): #for root, dirs, files in os.walk(rbp24_path):
    if root.endswith('test'):

        print(f"Processing file: {root}")

        positives_bed = root + "/positives.bed"
        negatives_bed = root + "/negatives.bed"
 
        original_df = pd.read_csv(root + "/original.tsv.gz", delimiter='\t', index_col=0, compression='gzip')

        negatives_df = original_df[original_df.label == 0]
        negatives_len = len(negatives_df)
        positives_df = original_df[original_df.label == 1]
        positives_len = len(positives_df)


        distributed_pos_df = pick_distributed_positives(negatives_df, positives_len, positives_bed, regions_bed, regions_tsv)
        distributed_neg_df = pick_distributed_negatives(positives_df, negatives_len, positives_bed, regions_bed, regions_tsv)
        random_pos_df = pick_random_positives(negatives_df, positives_len, random_transcripts_path)

        distributed_pos_df.to_csv(root + "/distributed_positives.tsv.gz", sep="\t", compression="gzip")
        distributed_neg_df.to_csv(root + "/distributed_negatives.tsv.gz", sep="\t", compression="gzip")
        random_pos_df.to_csv(root + "/random_positives.tsv.gz", sep="\t", compression="gzip")

