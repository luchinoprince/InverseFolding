import argparse
from Bio import SeqIO
import os
from biotite.structure.io import pdb
from biotite import structure, sequence


# PDBs do not necessarily contain all residues in a query used for MSA creation, so we create an explicit map here

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Create MSA encoding mapping')

    parser.add_argument('--output', default="msa_from_cath_pdbs.fasta")
    parser.add_argument('--pdb_dir', default="../dompdb")

    args = parser.parse_args()   

    pdb_list = list(os.listdir(args.pdb_dir))

    with open(args.output, 'w') as ofid:
        for pdb_file in pdb_list:
           
            with open(os.path.join(args.pdb_dir, pdb_file), 'r') as f:
                pdbf = structure.io.pdb.PDBFile.read(f)

            atom_array = structure.io.pdb.get_structure(pdbf, model=1)
            residue_ids, amino_acids = structure.get_residues(atom_array)

            amino_acids = [(a if a in sequence.ProteinSequence._dict_3to1 else 'UNK') for a in amino_acids]

            seq = ''.join(list(map(sequence.ProteinSequence.convert_letter_3to1, amino_acids)))

            ofid.write(f'>{pdb_file}\n')
            ofid.write(f'{seq}\n')




