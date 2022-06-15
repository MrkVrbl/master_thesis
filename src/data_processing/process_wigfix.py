import os
import re
from pathlib import Path
import numpy as np

chrom_sizes_path = '/home/mrkvrbl/Diplomka/Data/reference/referecnce_genome/hg19_chrom.sizes'
cons_wigs_path = Path('/home/mrkvrbl/Diplomka/Data/reference/conservation')
conservation_arrays_path = Path('/home/mrkvrbl/Diplomka/Data/reference/conservation/arrays')


def wigfix_arr_with_scores(cons_file, chroms_len_dict, name, file_size, outfile):
    print(f"    Creating {name} numpy file")
    """
    Loads conservation file in .wigFix format and transforms it into np.array spanning whole chromosome.
    Regions without conservation scores are set up to zero.
    
    args:
    cons_file - pathlib Path object to the conservation wigFix file
    chrom_len_dict - dictionary containing chromosome names and their lengths originaly downloaded as .sizes file
    name - wigFix filename without extension
    file_size - number of lines in the wigFix file
    outfile - pathlib Path to the output folder
    """ 
    # Initiate the np.array
    chrom_cons_array = np.zeros(chroms_len_dict[re.compile(r'chr[0-9A-Z]+').findall(str(cons_file))[0]]) 
    
    # Extract coordinates and conservation scores for all the intervals in conservation reference file 
    with open(cons_file, 'r') as inf:

        final_bedlike_strings = ''

        for i, line in enumerate(inf):
            if i == 0 and not line.startswith('fixedStep'):
                print('wrong format, stop')
                break

            elif i == 0 and line.startswith('fixedStep'):
                cons_scores_interval = []
                raw_header_els = line.strip().split(' ')[1:]
                header_els = [x.split('=')[1] for x in raw_header_els]
                chrom = header_els[0]; start = int(header_els[1]); step = int(header_els[2])
                new_start = start - 1
                counter = new_start

            elif line.startswith('fixedStep'):
                chrom_cons_array[new_start:counter] = np.array(cons_scores_interval)
                cons_scores_interval = []
                raw_header_els = line.strip().split(' ')[1:]
                header_els = [x.split('=')[1] for x in raw_header_els]
                chrom = header_els[0]; start = int(header_els[1]); step = int(header_els[2])
                new_start = start - 1
                counter = new_start

            elif i == file_size-1:
                score = float(line.strip())
                cons_scores_interval.append(score)
                chrom_cons_array[new_start:counter+step] = np.array(cons_scores_interval)
               
            else:
                score = float(line.strip())
                cons_scores_interval.append(score)
                counter += step
    print(f'{name}\'s scores extraction finished!')
    
    # Save np.array containing all conservation scores of the particular chromosome 
    return np.save(outfile, chrom_cons_array)


def blocks(cons_file, size=65536):
    """
    Memory efficient way to count number of lines in a huge file
    """
    while True:
        b = cons_file.read(size)
        if not b: break
        yield b


def process_wigfix(cons_wigs_path, chrom_sizes_path, conservation_arrays_path):
    print(f"Processing wigfix: {cons_wigs_path}")

    chrom_names = [re.compile(r'chr[0-9A-Z]*').findall(str(cons_file))[0] for cons_file in cons_wigs_path.glob('*.wigFix')]

    chroms_len_dict = {}
    with open(chrom_sizes_path, 'r') as sizes:
        for i, line in enumerate(sizes):
            chrom, size = line.strip().split('\t')
            if chrom in chrom_names:
                chroms_len_dict[chrom] = int(size)
                
    for cons_file in cons_wigs_path.glob('*.wigFix'):
        pattern = re.compile(r'chr[0-9A-Z]*.[0-9A-Za-z]*')
        name = pattern.findall(str(cons_file))[0]
        outfile = conservation_arrays_path / f'{name}.scores.npy'

        if os.path.isfile(outfile):
            print(f"File {outfile} already exists. Skipping...")
            continue
        else:
            with open(cons_file, "r", encoding="utf-8", errors='ignore') as f:
                file_size = sum(bl.count("\n") for bl in blocks(f))

            wigfix_arr_with_scores(cons_file, chroms_len_dict, name, file_size, outfile)


conservation_arrays_path.mkdir(exist_ok=True, parents=True)
process_wigfix(cons_wigs_path, chrom_sizes_path, conservation_arrays_path)