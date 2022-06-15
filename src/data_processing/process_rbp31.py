import os
import gzip
from pathlib import Path
from Bio import SeqIO
import pybedtools
from utils import concat_to_tsv, adjust_window


rbp31_path = "/home/mrkvrbl/Diplomka/Data/rbp31"
ref_fasta_path = '/home/mrkvrbl/Diplomka/Data/reference/referecnce_genome/hg19_latest.fa'
out_path = Path("/home/mrkvrbl/Diplomka/Data/rbp31/processed")
out_path.mkdir(exist_ok=True)
window_size = 101


def extract_coords(record):
    coords = record[0].split(",")
    return "\t".join([coords[0][1:]] + coords[2:])

def write_bed(file_out, records):
    with open(file_out, "w") as fout:
        for record in records:
            fout.write(record + "\n")

def map_bed(bed_files, ref_fasta, window_size):
    # pybedtool wrapper to call adjust_window and map .bed files to reference fasta, returns bed file with coordinates and sequences
    print(f"    Mapping to reference fasta...")
    for bed_file in bed_files:
        bed_file_out = bed_file[:-3] + "tsv"
        entries = pybedtools.BedTool(bed_file).sort().each(adjust_window, window_size).sequence(fi=ref_fasta, s=True, tab=True)
        entries.save_seqs(bed_file_out)


for root, dirs, files in os.walk(rbp31_path):

    if files:
        path_to_file = str(root) + "/sequences.fa.gz"
        pos_bed_file = root + "/positives.bed"
        neg_bed_file = root + "/negatives.bed"
        bed_files = [pos_bed_file, neg_bed_file]
        coords = []
        positives = []
        negatives = []

        records = [record.description.split(";") for record in SeqIO.parse(gzip.open(path_to_file, "rt"), 'fasta')]

        for record in records:
            if record[1] == ' class:1':
                pos_coords = extract_coords(record)
                positives.append(pos_coords)
            else:
                neg_coords = extract_coords(record)
                negatives.append(neg_coords)

        write_bed(pos_bed_file, positives)
        write_bed(neg_bed_file, negatives)
        map_bed(bed_files, ref_fasta_path, window_size)
        concat_to_tsv(root)
"""

#extract seqs
for root, dirs, files in os.walk(rbp31_path):

    if files:
        print(root)
        path_to_file = str(root) + "/sequences.fa.gz"
        file_out = str(root) + "/extracted.tsv"
        records = [[str(record.id), str(record.seq), record.description.split(";")[-1][-1]] for record in SeqIO.parse(gzip.open(path_to_file, "rt"), 'fasta')]

        with open(file_out, "w") as fout:
            for record in records:
                fout.write("\t".join(record) + "\n")

"""

