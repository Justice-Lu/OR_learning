import pandas as pd 
import numpy as np 
import sys 
import os 
import copy

import rmsd_functions as rmsd 

"""
The goal of this script is to convert all the tmalgned pdb coordinates into a fitted voxel 
Additionally, filter out the coordinates where the voxel positions are either empty or shared by all of the aligend ORs. 
"""

tmaligned_df = pd.read_pickle('../AF_files/dict_tmaligned.pkl')

def create_voxel(coords, size, resolution, spacer = [0,0,0]):
    # Initialize the voxel
    voxel = np.zeros(size, dtype=int)
    # Compute the indices of the coordinates in the voxel
    indices = np.floor((coords - spacer) / resolution).astype(int)
    # Set the values of the voxel
    voxel[indices[:,0], indices[:,1], indices[:,2]] = 1
    return voxel

resolution = 1

# Find the maximum extent of all the proteins
max_extent = np.max([np.max(tmaligned_df[Olfr]['coord'], axis=0) - \
                     np.min(tmaligned_df[Olfr]['coord'], axis=0) \
                     for Olfr in tmaligned_df], axis=0)
min_spacer = np.min([np.min(tmaligned_df[Olfr]['coord'], axis=0) \
                     for Olfr in tmaligned_df], axis=0)

size = np.ceil((max_extent - min_spacer )/ resolution).astype(int)

voxel_list = []
Olfr_order = []
# Loop through each protein's coordinates and create a voxel
sorted_keys = sorted(tmaligned_df.keys())
for Olfr in sorted_keys:
    voxel = create_voxel(tmaligned_df[Olfr]['coord'], 
                         size, resolution, 
                         spacer = min_spacer)
    # Save the voxel to a file or do other processing as needed
    Olfr_order.append(Olfr)
    voxel_list.append(voxel)
    
    
# flattens voxel via ravel function 
flat_voxel = []
for v in voxel_list:
    flat_voxel.append(np.ravel(v))
    
    
from tqdm import tqdm

with tqdm(total=len(flat_voxel[0])) as pbar:
    positive_space = np.nonzero(flat_voxel[0])[0]
    for v in flat_voxel: 
        new_values = np.setdiff1d(np.nonzero(v)[0], positive_space)
        if np.any(new_values): 
            positive_space = np.concatenate((positive_space, new_values))
# Arrange it to ascending order 
positive_space = np.sort(positive_space)

np.save('./voxel_info/pos-space_voxRes1.npy', positive_space)            
            


