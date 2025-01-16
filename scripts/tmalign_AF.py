
"""
Since tmalign takes a significantly longer time compared to kabasch. Use this script align and save files overnight 
"""

import numpy as np 
import pandas as pd 
import os 
import pickle
import copy
import sys 

# Function to load pickle data
def load_pickle(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except (pickle.PickleError, IOError):
        return {}

# Function to save pickle data
def save_pickle(data, filename):
    old_data = load_pickle(filename)    
    old_data.update(data)
    with open(filename, 'wb') as file:
        pickle.dump(old_data, file)
        

sys.path.insert(0, '/mnt/af3ff5c3-2943-4972-8c3a-6b98174779b7/Justice/OR_learning/utils')
import rmsd_functions as rmsd 

# Path to all AF Olfr directories
PDB_PATH = '/mnt/af3ff5c3-2943-4972-8c3a-6b98174779b7/Justice/AF_files/OR_AF_pdb/'
data_file = '/mnt/af3ff5c3-2943-4972-8c3a-6b98174779b7/Justice/AF_files/dict_tmaligned.pkl'
align_pdb_Olfr = 'Olfr1377'
aligned_Olfr = []

print("===================== Loading pdb files =====================", flush=True)
if os.path.isfile(data_file):
    data = load_pickle(data_file)
    aligned_Olfr += list(data.keys())
else:
    data = {}
    # Read in ref Olfr 
    align_ref = align_pdb_Olfr+'_0'
    data[align_ref] = {}
    data[align_ref]["atom"], data[align_ref]["coord"], \
    data[align_ref]["resid"], data[align_ref]['amino_acid'] = rmsd.get_coordinates_pdb(os.path.join(PDB_PATH,align_pdb_Olfr,'ranked_0.pdb'))


unaligned_Olfr = []
for num_i, olfr in enumerate(os.listdir(PDB_PATH)):
    print(f'=== {num_i} / {len(os.listdir(PDB_PATH))} ===', flush=True)
    
    # Read in ref Olfr
    new_data = {}
    align_ref = align_pdb_Olfr+'_0'
    new_data[align_ref] = {}
    new_data[align_ref]["atom"], new_data[align_ref]["coord"], \
    new_data[align_ref]["resid"], new_data[align_ref]['amino_acid'] = rmsd.get_coordinates_pdb(os.path.join(PDB_PATH,align_pdb_Olfr,'ranked_0.pdb'))
    
    ranked_pdb_files = [i for i in os.listdir(os.path.join(PDB_PATH, olfr)) if 'ranked_' in i ]
    ranked_pdb_files = np.sort(ranked_pdb_files)    
    # Read in Olfr to align 
    for rank, file in enumerate(ranked_pdb_files): 
        try:
            dict_name = '_'.join([olfr, str(rank)])
            new_data[dict_name] = {}
            new_data[dict_name]["atom"], new_data[dict_name]["coord"], \
            new_data[dict_name]["resid"], new_data[dict_name]['amino_acid'] = rmsd.get_coordinates_pdb(os.path.join(PDB_PATH,olfr,file))
            
            unaligned_Olfr.append(dict_name)
        except:
            print(file + ' NOT read')
            
    print("{} Olfr loaded".format(len(new_data)), flush=True)

    # Remove H atom, as only relaxed model contains H. (ranked_0.pdb)
    new_data = rmsd.remove_H(new_data)
    # Extract backbone specifically for tmalignment
    backbone = copy.deepcopy(new_data)
    backbone = rmsd.get_backbone(backbone)
    # Only use aa30-300 for alignment. roughly TM domain
    backbone = rmsd.trim_resid(backbone, start=30, end=300)

    # print("Quality Control: Dropping small Olfr lesser than 285 resid")
    # small_Olfr = []
    # for Olfr in backbone:
    #     if backbone[Olfr]['resid'].max() < 285:
    #         small_Olfr.append(Olfr)
    # for Olfr in small_Olfr: 
    #     data.pop(Olfr)
    #     backbone.pop(Olfr)
    # print("{} considerred short \n{} Olfr left".format(len(small_Olfr), len(data)-len(small_Olfr)))

    # print("===================== Begin tmalign rotation =====================")
    aligned_data = rmsd.tmalign_data(backbone, align=align_ref, rotate_data = new_data, skip_data = aligned_Olfr)
    # print("Rotation alignment done")
    
    # Update aligned_Olfr
    aligned_Olfr += unaligned_Olfr
    
    # Joins the new dictionary item with the old data
    data.update(aligned_data)
    save_pickle(data, data_file)






