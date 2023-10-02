import os
import numpy as np
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import Select
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBList


# Define the amino acid alphabet and its encoding
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_ENCODING = np.eye(len(AA_ALPHABET))

class AlphaFoldSelect(Select):
    def accept_residue(self, residue):
        # Accepts all residues except water and hetero atoms
        if residue.get_resname() in {"HOH", "WAT"} or residue.id[0].strip() != '':
            return 0
        else:
            return 1

def one_hot_encode_alphafold_pdb(pdb_file):
    """One-hot encodes an AlphaFold PDB file"""
    # Load the PDB file
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    # Get the amino acid sequence
    ppb = PPBuilder()
    pp = ppb.build_peptides(structure)
    sequence = ''.join([str(p.get_sequence()) for p in pp])

    # One-hot encode the amino acid sequence
    encoded_seq = []
    for aa in sequence:
        encoded_seq.append(AA_ENCODING[AA_ALPHABET.index(aa)])
    encoded_seq = np.array(encoded_seq)

    # Return the encoded sequence
    return encoded_seq

def pdb_to_aa_and_ss(pdb_path, simplify_ss=False):
    
    """
    This function reads in PDB files generated from Alphafold 
    Returns aminoacid sequence and the corresponding secondary structures in string format 
    simplify_ss is used to conver the 8 varied DSSP secondary structure to simple 
    Coil, Helix, and Extended
    """
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_path)

    # Compute the secondary structure features
    model = structure[0]
    dssp = DSSP(model, pdb_path)
    dssp_dict = dssp.property_dict

    # Print the secondary structure features for each residue
    aa = ""
    ss = ""
    for residue in range(len(dssp)):
        residue_key = list(dssp.keys())[residue]
        aa += dssp[residue_key][1]
        ss += dssp[residue_key][2]
    #     phi, psi = dssp_dict[residue_key][3:5]
    #     print('{}{}: SS={}, phi={}, psi={}'.format(aa, res_id[3][1], ss, phi, psi))
    """
        The DSSP codes for secondary structure used here are:
        =====     ====
        Code      Structure
        =====     ====
        H         Alpha helix (4-12)
        B         Isolated beta-bridge residue
        E         Strand
        G         3-10 helix
        I         Pi helix
        T         Turn
        S         Bend
        -         None
        =====     ====
    """
    # if desired, convert DSSP's 8-state assignments into 3-state [C - coil, E - extended (beta-strand), H - helix]
    if simplify_ss:
        ss = ss.replace('-', 'C')
        ss = ss.replace('I', 'C')
        ss = ss.replace('T', 'C')
        ss = ss.replace('S', 'C')
        ss = ss.replace('G', 'H')
        ss = ss.replace('B', 'E')
    return aa, ss 




# Define a dictionary mapping amino acids to integers
aa_dict = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19
}

# Define a function to convert an amino acid sequence to a kmer matrix
def seq_to_kmer_matrix(seq, k, max_len):
    # Initialize the matrix with zeros
    matrix = np.zeros((max_len, 20**k))

    # Convert the sequence to a list of integers
    seq_int = [aa_dict.get(aa, aa_dict['X']) for aa in seq]

    # Iterate over kmers of length k in the sequence
    for i in range(max_len - k + 1):
        kmer_int = 0
        for j in range(k):
            kmer_int += seq_int[i+j] * 20**(k-j-1)
        matrix[i, kmer_int] = 1

    return matrix
