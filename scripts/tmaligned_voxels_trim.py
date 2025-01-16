import pandas as pd 
import numpy as np 
import sys 
import os 
import copy

import rmsd_functions as rmsd 
import voxel_functions as vx_func


"""
The goal of this script is to convert all the tmalgned pdb coordinates into a fitted voxel 
Additionally, filter out the coordinates where the voxel positions are either empty or shared by all of the aligend ORs. 
"""

tmaligned_df = pd.read_pickle('../AF_files/dict_tmaligned.pkl')


resolution_set = 1

from tqdm import tqdm

voxel_list, voxel_shape, Olfr_order = vx_func.create_voxel(tmaligned_df, 
                                                     resolution = resolution_set, 
                                                      fill_radius=True)    
    
# flattens voxel via ravel function 
flat_voxel = []
for v in voxel_list:
    flat_voxel.append(np.ravel(v))
    
    
positive_space = np.nonzero(flat_voxel[0])[0]
for v in tqdm(flat_voxel): 
    new_values = np.setdiff1d(np.nonzero(v)[0], positive_space)
    if np.any(new_values): 
        positive_space = np.concatenate((positive_space, new_values))
# Arrange it to ascending order 
positive_space = np.sort(positive_space)
flat_voxel_trimmed = [np.array(v)[positive_space].copy() for v in flat_voxel]

np.save('./voxel_info/Res1/voxel_shape_Res1_fillradius.npy', voxel_shape)
np.save('./voxel_info/Res1/flat_voxel_trimmed_Res1_fillradius.npy', flat_voxel_trimmed)
np.save('./voxel_info/Res1/pos-space_voxRes1_fillradius.npy', positive_space)            
            


