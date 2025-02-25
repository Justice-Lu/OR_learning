import os 
import sys 
from tqdm import tqdm 

OR_LEARNING_PATH = os.path.join(os.getcwd().split('OR_learning')[0], 'OR_learning/')
sys.path.insert(0, os.path.join(OR_LEARNING_PATH, 'utils/'))

import BindingCavity_functions as bc 

AF2_PATH = '/mnt/data2/Justice/AF_files/AF_tmaligned_pdb/'
pdb_files = os.listdir(AF2_PATH)

reference_pdb = os.path.join(AF2_PATH, "Or51E2_Mol2.3_Olfr78_Psgr_tmaligned.pdb")
aligned_pdbs = [os.path.join(AF2_PATH, _pdb) for _pdb in pdb_files]

aligned_pairs = bc.generate_sequence_alignment_pairs_fromPDB(
    reference_pdb,
    tqdm(aligned_pdbs),
    load_pdb_fn=bc.load_pdb_coordinates,  # Replace with your PDB-loading function
    gap_penalty=5
)

alignment = bc.union_gaps_with_consistency(aligned_pairs)

import pickle 
with open(os.path.join(OR_LEARNING_PATH, 'files/binding_cavity/dict_bc_SeqAlignment_Full.pkl'), 'wb') as f:
    pickle.dump(alignment, f)