import gzip
import shutil
import os
from pathlib import Path

dir_path = Path("/home/mrkvrbl/Diplomka/Data/reference/conservation_scores")
#file_path_out = Path("/home/mrkvrbl/Diplomka/Data/reference/referecnce_genome/hg19_latest.fa")


for file in os.listdir(dir_path):
    file_path_in = str(dir_path) + "/" + file
    if file_path_in.endswith(".wigFix"):
        file_path_out = str(file_path_in) + ".gz"
        with open(file_path_in, 'rb') as f_in:
            with gzip.open(file_path_out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    if str(file_path_in).endswith(".wigFix"):
        os.remove(file_path_in)

    print(f"{file_path_out} processed")

"""

with gzip.open(file_path_out, 'rb') as f_out:
    lines = f_out.read()
    print(f"content:{lines}")


for file in os.listdir(dir_path):
    if file.endswith(".fa"):
        file_path = str(dir_path) + "\\" + file
        os.remove(file_path)


with gzip.open(file_path_in, 'rt') as f_in:
    with open(file_path_out, 'w') as f_out:
        shutil.copyfileobj(f_in, f_out)

"""