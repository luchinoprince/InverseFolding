from ast import BinOp
from genericpath import isdir
import pickle as pkl
import os
import argparse
import shutil
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import random



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--msa_dir', type=str, default=os.path.expanduser('~/cloud_share/cath_alignments')) 
    parser.add_argument('--encoding_dir', type=str, default=os.path.expanduser('~/Downloads/encoded_proteins'))
    parser.add_argument('--output_dir', type=str, default='toy_data')
    parser.add_argument('--num_toys', type=int, default=50)
    parser.add_argument('--max_gap_frac', type=float, default=0.3)

    args = parser.parse_args()


    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)


    msa_files =  list(filter(lambda file: file.endswith('a3m'), os.listdir(args.msa_dir)))
    encoding_files = list(filter(lambda file: file.endswith('pkl'), os.listdir(args.encoding_dir)))

    files_done = 0
    random.shuffle(encoding_files)
    for encoding_file in encoding_files:

   
        cathid = encoding_file.split('_')[2].split('.')[0]
        pdbid = cathid[:4]
        chain = cathid[4]

        msa_file = "{}_{}.a3m".format(pdbid, chain)

        if msa_file not in msa_files:
            continue

        

        seq_records = list(SeqIO.parse(os.path.join(args.msa_dir, msa_file), 'fasta'))

        assert seq_records[0].id == 'query'

        pdb_seq = str(seq_records[0].seq)

        assert pdb_seq.isupper()

        with open(os.path.join(args.encoding_dir, encoding_file), 'rb') as f:
            encoding = pkl.load(f)
        
        cath_seq = encoding['Native_Seq']
        cath_start = pdb_seq.find(cath_seq)

        if cath_start == -1:
            continue

        print(msa_file)
        
        cath_seq_len = len(cath_seq)

        seq_records_aligned = []
        for seq_record in seq_records:
            seq = str(seq_record.seq)[cath_start:]
            matchstates = [s.isupper() or s == '-' for s in seq]
            cath_end = np.cumsum(matchstates).searchsorted(cath_seq_len)+1

            seq = seq[:cath_end]
            if seq_record.id == 'query':
                assert ''.join([s for s in seq if s.isupper()]) == cath_seq

            gap_fraction = [c == '-' for c in seq].count(True) / len(seq)

            if gap_fraction > args.max_gap_frac:
                continue

            seq_records_aligned.append(SeqIO.SeqRecord(Seq(seq), id=seq_record.id))
                    
        output_file = os.path.join(args.output_dir, encoding_file.replace("pkl", "fasta"))

        SeqIO.write(seq_records_aligned, output_file, 'fasta')
        shutil.copy(os.path.join(args.encoding_dir, encoding_file), os.path.join(args.output_dir, encoding_file))
        
        files_done += 1
        if files_done >= args.num_toys:
            break

