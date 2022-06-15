import pybedtools
from utils import intersection, adjust_window

transcripts = "/home/mrkvrbl/Diplomka/Data/outputs/transcripts_rbp24.bed"
reference_fasta = "/home/mrkvrbl/Diplomka/Data/reference/referecnce_genome/hg19_latest.fa"

introns_in = "/home/mrkvrbl/Diplomka/Data/reference/region_types/introns.bed.gz"
exons_in = "/home/mrkvrbl/Diplomka/Data/reference/region_types/exons.bed.gz"
utr5_in = "/home/mrkvrbl/Diplomka/Data/reference/region_types/5utrs.bed.gz"
utr3_in = "/home/mrkvrbl/Diplomka/Data/reference/region_types/3utrs.bed.gz"

introns_out = "/home/mrkvrbl/Diplomka/Data/reference/region_types/introns_intersections.tsv"
exons_out = "/home/mrkvrbl/Diplomka/Data/reference/region_types/exons_intersections.tsv"
utr3_out = "/home/mrkvrbl/Diplomka/Data/reference/region_types/utr3_intersections.tsv"
utr5_out = "/home/mrkvrbl/Diplomka/Data/reference/region_types/utr5_intersections.tsv"

def intersection(a, b):
    a = pybedtools.BedTool(a)
    b = pybedtools.BedTool(b)

    return a.intersect(b)

def map_bed(bed_file, ref_fasta, window_size, out_path):
    # pybedtool wrapper to call adjust_window and map .bed files to reference fasta, returns bed file with coordinates and sequences
    print(f"    Mapping to reference fasta...")
    entries = bed_file.sort().each(adjust_window, window_size).sequence(fi=ref_fasta, s=True, tab=True)
    entries.save_seqs(out_path)

introns_transcripts = intersection(transcripts, introns_in)
exons_transcripts = intersection(transcripts, exons_in)
utr3_transcripts = intersection(transcripts, utr3_in)
utr5_transcripts = intersection(transcripts, utr5_in)

map_bed(introns_transcripts, reference_fasta, 128, introns_out)
map_bed(exons_transcripts, reference_fasta, 128, exons_out)
map_bed(utr3_transcripts, reference_fasta, 128, utr3_out)
map_bed(utr5_transcripts, reference_fasta, 128, utr5_out)

