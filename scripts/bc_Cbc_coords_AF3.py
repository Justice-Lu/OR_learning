import os 
import sys
import pandas as pd 
import numpy as np 

OR_LEARNING_PATH = os.path.join(os.getcwd().split('OR_learning')[0], 'OR_learning/')
sys.path.insert(0, os.path.join(OR_LEARNING_PATH, 'utils/'))

import BindingCavity_functions as bc 



"""
Re-Filtering the cavity and residue by the cavity that overlaps with canonical_bc_coords

canonical_bc_coords is previously defined via AF3 structure overlapping cavity with residues inside. 
See bc_AF3_CBC.ipynb

"""

cav_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/human/AF3_dict_bc_pyKVcav_tmaligned.pkl')
res_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/human/AF3_dict_bc_pyKVres_tmaligned.pkl')

canonical_bc_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/canonical_bc_coords.pkl')

Cbc_cav_coords = {}
Cbc_res_coords = {}
for _OR in cav_coords: 
    Cbc_cav_coords[_OR], Cbc_res_coords[_OR] = bc.filter_coordinates_within_cavity(canonical_bc_coords, 
                                                                     cav_coords[_OR], 
                                                                     res_coords[_OR], 
                                                                     filter_cutoff=0.3)

# Remove Olfr entries with no detected cavity / residue after filtering
empty_olfr = [_olfr for _olfr in Cbc_cav_coords if len(Cbc_cav_coords[_olfr]) == 0]
for _olfr in empty_olfr: 
    del Cbc_cav_coords[_olfr]
    del Cbc_res_coords[_olfr]
    
# Filter by DL_OR only ORs
# Olfr_DL = np.load('/mnt/data2/Justice/OR_learning/files/Olfr_DL.npy', allow_pickle=True).item()

Cbc_cav_coords = {_or : _value for _or, _value in Cbc_cav_coords.items()}
Cbc_res_coords = {_or : _value for _or, _value in Cbc_res_coords.items()}

import pickle 
with open('/mnt/data2/Justice/OR_learning/files/binding_cavity/human/AF3_dict_Cbc_cav_coords.pkl', 'wb') as f:
    pickle.dump(Cbc_cav_coords, f)
with open('/mnt/data2/Justice/OR_learning/files/binding_cavity/human/AF3_dict_Cbc_res_coords.pkl', 'wb') as f:
    pickle.dump(Cbc_res_coords, f)
