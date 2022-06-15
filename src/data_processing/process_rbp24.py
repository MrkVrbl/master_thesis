import os
import site
import fnmatch
import gzip
from pathlib import Path
import pybedtools
from sklearn.utils import shuffle
from utils import concat_to_tsv, adjust_window
#from data_processing.utils import compute_ss, map_cons #src.data_processing.
site.addsitedir('/usr/local/lib/python3/site-packages/')  # Always appends to end


def make_train_test_dir(path, rbps, i):
    rbp_name = rbps[i].split(".")[0]
    protein_path = Path(os.path.join(path, rbp_name))
    train_path = Path(os.path.join(path, (rbp_name + "/train")))
    test_path = Path(os.path.join(path, (rbp_name + "/test")))

    protein_path.mkdir(exist_ok=True)
    train_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)
    return train_path, test_path, protein_path


def create_bed_file(rbp24_path, rbp_file, processed_rbp_path):
    #Extracts chromosomes and coordinates from fasta file, returns .bed file
    label = rbp_file.split(".")[2]
    bed_file_path = str(processed_rbp_path) + '/' + label + '.bed'
    fasta_file_path = str(rbp24_path) + '/' + rbp_file
    with gzip.open(fasta_file_path, 'rt') as fasta_file, open(bed_file_path, 'w') as bed_file:
        for line in fasta_file:
            if line.startswith('>'):
                fields = line.split(';')[1].split(',')
                bed_file.write('\t'.join(fields))
    return bed_file


def create_bed_files(rbp24_path, rbps, test_path, train_path, i):
    print(f"    Creating bed files...")
    bed_files = []
    for j in range(4):
        if fnmatch.fnmatch(rbps[i+j], '*.ls.*'):
            bed_files.append(create_bed_file(rbp24_path, rbps[i+j], test_path))
        elif fnmatch.fnmatch(rbps[i+j], '*.train.*'):
            bed_files.append(create_bed_file(rbp24_path, rbps[i+j], train_path))
        else:
            print("Skipping file: {rbps[i+j]}")
    return bed_files


def get_sequences(path):
    sequences = []
    with gzip.open(path, "rb") as f_in:
        lines = f_in.readlines()
        for line in lines:
            line = line.decode("ascii")
            if not line.startswith(">"):
                line = line.lower().strip()
                line = "".join([char if char != "u" else "t" for char in line]) #translate
                sequences.append(line)
    return sequences


def save_data(data, path):
    with open(path, "w") as f_out:
        for entry in data:
            f_out.write(entry[0] + "\t" + str(entry[1]) + "\n")
    return


def tsv_from_fasta(rbp24_path, rbps, test_path, train_path, i):
    rbp_file_paths = [Path(str(rbp24_path) + "/" + rbp) for rbp in rbps[i:i+4]]
    train_file_paths = sorted([file for file in rbp_file_paths if fnmatch.fnmatch(file, '*.train.*')])
    test_file_paths = sorted([file for file in rbp_file_paths if fnmatch.fnmatch(file, '*.ls.*')])
    
    train_positives = get_sequences(train_file_paths[1])
    train_positives = [[seq, 1] for seq in train_positives]
    train_negatives = get_sequences(train_file_paths[0])
    train_negatives = [[seq, 0] for seq in train_negatives]

    test_positives = get_sequences(test_file_paths[1])
    test_positives = [[seq, 1] for seq in test_positives]
    test_negatives = get_sequences(test_file_paths[0])
    test_negatives = [[seq, 0] for seq in test_negatives]

    train = shuffle(train_negatives + train_positives)
    test = shuffle(test_negatives + test_positives)

    save_data(train, str(train_path) + "/original.tsv")
    save_data(test, str(test_path) + "/original.tsv")


def map_bed(bed_files, ref_fasta, window_size):
    # pybedtool wrapper to call adjust_window and map .bed files to reference fasta, returns bed file with coordinates and sequences
    print(f"    Mapping to reference fasta...")
    for bed_file in bed_files:
        bed_file_out = bed_file.name[:-3] + "tsv"
        entries = pybedtools.BedTool(bed_file.name).sort().each(adjust_window, window_size).sequence(fi=ref_fasta, s=True, tab=True)
        entries.save_seqs(bed_file_out)


def full_preprocessing(rbp24_path, out_path, ref_fasta_path, window_size):
    #extract chromosome coordinates of reads from the fasta files, yeald sequences of specific length by mapping the coordinates to the reference genome.
    rbps = sorted(os.listdir(rbp24_path))
    print(f'Number of files to process: {len(rbps)}')
    print("...")

    for i in range(0, len(rbps), 4):
        train_path, test_path, protein_path = make_train_test_dir(out_path, rbps, i)
        bed_files = create_bed_files(rbp24_path, rbps, test_path, train_path, i)
        map_bed(bed_files, ref_fasta_path, window_size)
        concat_to_tsv(protein_path)
        #compute_ss(bed_files)
        #map_cons(bed_files, conservation_arrays_path)
    print("DONE")


def extract_seqs(rbp24_path, out_path_extracted):
    #extract sequences from protein fasta files.
    rbps = sorted(os.listdir(rbp24_path))
    print(f'Number of files to process: {len(rbps)}')
    print("...")

    for i in range(0, len(rbps), 4):
        print(f"processing rbp: {rbps[i].split('.')[0]}")
        train_path, test_path, _ = make_train_test_dir(out_path_extracted, rbps, i)
        tsv_from_fasta(rbp24_path, rbps, test_path, train_path, i)
    print("DONE")


rbp24_path = Path('/home/mrkvrbl/Diplomka/Data/rbp24/raw')
out_path = Path('/home/mrkvrbl/Diplomka/Data/rbp24/processed')
out_path_extracted = Path('/home/mrkvrbl/Diplomka/Data/rbp24/extracted_seqs')
ref_fasta_path = '/home/mrkvrbl/Diplomka/Data/reference/referecnce_genome/hg19_latest.fa'
window_size = 101

out_path.mkdir(exist_ok=True)
out_path_extracted.mkdir(exist_ok=True)

full_preprocessing(rbp24_path, out_path, ref_fasta_path, window_size)
#extract_seqs(rbp24_path, out_path_extracted)