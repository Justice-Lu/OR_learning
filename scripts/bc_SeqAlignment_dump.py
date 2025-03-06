import os 
import sys 
import numpy as np 
from tqdm import tqdm 

OR_LEARNING_PATH = os.path.join(os.getcwd().split('OR_learning')[0], 'OR_learning/')
sys.path.insert(0, os.path.join(OR_LEARNING_PATH, 'utils/'))

import SequenceAlignment_functions as sa
import pdb_functions as pu

AF2_PATH = '/mnt/data2/Justice/AF_files/AF_tmaligned_pdb/'
pdb_files = os.listdir(AF2_PATH)

# reference_pdb = os.path.join(AF2_PATH, "Or51E2_Mol2.3_Olfr78_Psgr_tmaligned.pdb")
# aligned_pdbs = [os.path.join(AF2_PATH, _pdb) for _pdb in pdb_files]

# aligned_pairs = bc.generate_sequence_alignment_pairs_fromPDB(
#     reference_pdb,
#     tqdm(aligned_pdbs),
#     load_pdb_fn=bc.load_pdb_coordinates,  # Replace with your PDB-loading function
#     gap_penalty=5
# )

# alignment = bc.union_gaps_with_consistency(aligned_pairs)

# Read in and align sequences first

pdb_files  = os.listdir('/mnt/data2/Justice/AF_files/AF_tmaligned_pdb/')
pdb_files  = [os.path.join('/mnt/data2/Justice/AF_files/AF_tmaligned_pdb/', _file) for _file in pdb_files]

Olfr_DL = np.load('/mnt/data2/Justice/OR_learning/files/Olfr_DL.npy', allow_pickle=True).item()
pdb_labels = [Olfr_DL.get(_pdb.split('/')[-1].split('_')[0], _pdb.split('/')[-1].split('_')[0]) for _pdb in pdb_files ]

ref_pdb = '/mnt/data2/Justice/OR_learning/files/TEST_modeller/AF2_Or51E2_tmaligned.pdb'

aligned_pairs = sa.generate_sequence_alignment_pairs_fromPDB(ref_pdb,
                                                               tqdm(pdb_files), 
                                                               pu.load_pdb_coordinates, 
                                                               labels = pdb_labels, 
                                                               gap_penalty=100)
alignment = sa.union_gaps_with_consistency(aligned_pairs)

import pickle 
with open(os.path.join(OR_LEARNING_PATH, 'files/binding_cavity/dict_SeqAlignment_Full_gp100.pkl'), 'wb') as f:
    pickle.dump(alignment, f)