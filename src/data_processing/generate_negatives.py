import pybedtools
from pathlib import Path
import pandas as pd
import glob
import os


INTRONS = "/home/mrkvrbl/Diplomka/Data/reference/region_types/introns.bed.gz"
EXONS = "/home/mrkvrbl/Diplomka/Data/reference/region_types/exons.bed.gz"
UTR5 = "/home/mrkvrbl/Diplomka/Data/reference/region_types/5utrs.bed.gz"
UTR3 = "/home/mrkvrbl/Diplomka/Data/reference/region_types/3utrs.bed.gz"
REPEATS = "/home/mrkvrbl/Diplomka/Data/reference/repeat_masker_hg19.gz"

rbp24_path = "/home/mrkvrbl/Diplomka/Data/rbp24/processed"


def intersection(a, b):

    a = pybedtools.BedTool(a)
    b = pybedtools.BedTool(b)

    return a.intersect(b, u=True, f=0.51, wa=True)


def sample(pos, neg):

    if len(pos) >= len(neg):
        return neg
    
    return neg.sample(n = max(1, len(pos)))

def generate_distrib(positives):

    #generate negaatives by shuffling the coordinates of possitives in each region separately
    introns = positives.shuffle(genome='hg19', chrom=True, noOverlapping=True, incl=INTRONS)
    exons = positives.shuffle(genome='hg19', chrom=True, noOverlapping=True, incl=EXONS)
    utr5 = positives.shuffle(genome='hg19', chrom=True, noOverlapping=True, incl=UTR5)
    utr3 = positives.shuffle(genome='hg19', chrom=True, noOverlapping=True, incl=UTR3)

    #intersect postivies with each region
    pos_intron_intersect = intersection(positives, INTRONS)
    pos_exon_intersect = intersection(positives, EXONS)
    pos_utr3_intersect = intersection(positives, UTR3)
    pos_utr5_intersect = intersection(positives, UTR5)
    pos_repeat_intersect = intersection(positives, REPEATS)

    # get repeat and non repeat sequences from each region of positive sequences
    pos_intron_repeat = intersection(pos_intron_intersect, pos_repeat_intersect)
    pos_intron_nonrepeat = pos_intron_intersect - pos_intron_repeat

    pos_exon_repeat = intersection(pos_exon_intersect, pos_repeat_intersect)
    pos_exon_nonrepeat = pos_exon_intersect - pos_exon_repeat

    pos_utr3_repeat = intersection(pos_utr3_intersect, pos_repeat_intersect)
    pos_utr3_nonrepeat = pos_utr3_intersect - pos_utr3_repeat

    pos_utr5_repeat = intersection(pos_utr5_intersect, pos_repeat_intersect)
    pos_utr5_nonrepeat = pos_utr5_intersect - pos_utr5_repeat

    # get repeat and nonrepeat regions of generated negative sequences
    neg_intron_repeat = intersection(introns, REPEATS)
    neg_intron_nonrepeat = introns - neg_intron_repeat

    neg_exon_repeat = intersection(exons, REPEATS)
    neg_exon_nonrepeat = exons - neg_exon_repeat

    neg_utr3_repeat = intersection(utr3, REPEATS)
    neg_utr3_nonrepeat = utr3 - neg_utr3_repeat

    neg_utr5_repeat = intersection(utr5, REPEATS)
    neg_utr5_nonrepeat = utr5 - neg_utr5_repeat


    #sample the generated sequences acording to the positive sequences distribution
    sampled_neg_intron_repeat = sample(pos_intron_repeat, neg_intron_repeat)
    sampled_neg_intron_nonrepeat = sample(pos_intron_nonrepeat, neg_intron_nonrepeat)

    sampled_neg_exon_repeat = sample(pos_exon_repeat, neg_exon_repeat)
    sampled_neg_exon_nonrepeat = sample(pos_exon_nonrepeat, neg_exon_nonrepeat)

    sampled_neg_utr3_repeat = sample(pos_utr3_repeat, neg_utr3_repeat)
    sampled_neg_utr3_nonrepeat = sample(pos_utr3_nonrepeat, neg_utr3_nonrepeat)

    sampled_neg_utr5_repeat = sample(pos_utr5_repeat, neg_utr5_repeat)
    sampled_neg_utr5_nonrepeat = sample(pos_utr5_nonrepeat, neg_utr5_nonrepeat)

    #put it together

    negatives = {"neg_intron_repeat":sampled_neg_intron_repeat, 
                "neg_intron_nonrepeat":sampled_neg_intron_nonrepeat,
                "neg_exon_repeat":sampled_neg_exon_repeat,
                "neg_exon_nonrepeat":sampled_neg_exon_nonrepeat, 
                "neg_utr5_repeat":sampled_neg_utr5_repeat, 
                "neg_utr5_nonrepeat":sampled_neg_utr5_nonrepeat, 
                "neg_utr3_repeat":sampled_neg_utr3_repeat, 
                "neg_utr3_nonrepeat":sampled_neg_utr3_nonrepeat}

    return negatives

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


def map_bed(path, negatives):

    for bed_file_name, bed_tool_obj in negatives.items():
        bed_file_out = path + "/" + bed_file_name + ".tsv"
        entries = bed_tool_obj.sort().each(adjust_window, 101).sequence(fi="/home/mrkvrbl/Diplomka/Data/reference/referecnce_genome/hg19_latest.fa", s=True, tab=True)
        entries.save_seqs(bed_file_out)

def get_dataframe(path):

    negatives_path = path + "/neg_*repeat.tsv"
    tsv_files = glob.glob(negatives_path)
    dfs = []
    for file in tsv_files:
        df = pd.read_csv(file,delimiter="\t", header=None, names=['id', 'seq'])
        df['label'] = 0
        dfs.append(df)
        os.remove(file)

    return pd.concat(dfs).sample(frac=1).reset_index(drop=True)


def get_gc_count_from_seq(seq):
    gc = 0
    count = 0

    if type(seq) == "str":
        seq = seq.lower()
        for char in seq:
            count += 1
            if char == "g" or char == "c":
                gc += 1
            
        return gc * 100 / count

    seqs = seq

    for seq in seqs:
        seq = seq.lower()
        for char in seq:
            count += 1
            if char == "g" or char == "c":
                gc += 1

    return gc * 100 / count

def drop_high_gc(neg_df, n):
    neg_df['GC'] = neg_df.seq.apply(lambda x: get_gc_count_from_seq(x))
    neg_df = neg_df.sort_values("GC", ascending=False).reset_index(drop=True)
    return neg_df.iloc[n:]




for root, dirs, files in os.walk(rbp24_path): #for root, dirs, files in os.walk(rbp24_path):
    if root.endswith('test'):

        print(f"Processing file: {root}")

        positives_bed = pybedtools.BedTool(Path(root + "/positives.bed"))
        original_df = pd.read_csv(root + "/original.tsv.gz", delimiter="\t", index_col=0, compression="gzip")
        pos_df = original_df[original_df.label == 1]

        negatives_dict = generate_distrib(positives_bed)
        map_bed(root, negatives_dict)
        neg_df = get_dataframe(root)

        neg_gc = get_gc_count_from_seq(neg_df.seq)
        pos_gc = get_gc_count_from_seq(pos_df.seq)

        #if neg_gc > pos_gc and len(neg_df) > len(pos_df):
        #    n = len(neg_df) - len(pos_df)
        #    neg_df = drop_high_gc(neg_df, n)
        
        dist_negatives_df = pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
        dist_negatives_df.to_csv(root + "/dist_neg_orig_pos.tsv.gz", sep="\t", compression="gzip")


