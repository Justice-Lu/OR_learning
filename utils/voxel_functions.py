import pandas as pd 
import numpy as np 
import os 
import copy
from tqdm import tqdm
import math 
import torch 

# DEPRECATED FUNCTIONS 
# """
# Takes in dict generated from reading PDB to generate information neede for making voxels. 

# dict = {'Olfr0' : {'atom': np.array([],
#                      'coord': np.array([],
#                      'resid': np.array(), 
#                      'amino_acid': np.array()
#                      },
#         'Olfr1' : ...
#         }
# """
# def PDB_voxel_info(PDB_dict, resolution = 1):
#     # Find the maximum extent of all the proteins
#     max_extent = np.max([np.max(PDB_dict[Olfr]['coord'], axis=0) - \
#                          np.min(PDB_dict[Olfr]['coord'], axis=0) \
#                          for Olfr in PDB_dict], axis=0)
#     min_spacer = np.min([np.min(PDB_dict[Olfr]['coord'], axis=0) \
#                          for Olfr in PDB_dict], axis=0)
#     voxel_shape = np.ceil((max_extent - min_spacer )/ resolution).astype(int)
    
#     return max_extent, min_spacer, voxel_shape
    


# """
# Create a voxel from pdb coordinates. 
# coords - PDB coordinate
# resolution - the resolution of the voxel in Å. (Scales up exponentially) 
# """
# def create_voxel(PDB_dict,  
#                  resolution, 
#                  fill_radius=False):
    
#     max_extend, min_spacer, voxel_shape = PDB_voxel_info(PDB_dict, resolution = resolution)
#     # Calculate the voxel shape based on the maximum and minimum coordinates
#     voxel_shape = voxel_shape + np.ceil((max_extend - min_spacer) / resolution).astype(int)
    
#     voxel_list = []    
#     Olfr_order = sorted(PDB_dict.keys())
#     for Olfr in tqdm(Olfr_order): 
#         # Initialize the voxel
#         voxel = np.zeros(np.array(list(voxel_shape) + [len(ATOM_ENCODING)]), dtype=int)
        
#         # Compute the indices of the coordinates in the voxel
#         # indices = np.floor((PDB_dict[Olfr]['coord'] - min_spacer) / resolution).astype(int)
        

#         # Set the values of the voxel based on radii
#         if fill_radius: 
#             for idx, index in enumerate(indices):
#                 voxel_index = tuple(index)
#                 radius = ATOM_RADIUS_DICT[PDB_dict[Olfr]['atom'][idx]]
#                 num_points = int(np.ceil(radius / resolution))
#                 assign_voxel(voxel, voxel_index, num_points)
#         else: 
#             # Previously does not discriminate between atoms
#             # voxel[indices[:,0], indices[:,1], indices[:,2]] = 1
            
#             # Get indice for each atom and assign OHE vector
#             for atom in ATOM_ENCODING.keys():
#                 atom_indice = PDB_dict[Olfr]['coord'][np.where(PDB_dict[Olfr]['atom'] == atom)]
#                 indices = np.floor((atom_indice - min_spacer) / resolution).astype(int)
#                 voxel[indices[:,0], indices[:,1], indices[:,2]] = ATOM_ENCODING[atom]
#                 # Line to check to prevent loss of resolution when scaled 
#                 # print(f'indices {len(indices)}... num_pos {len(np.argwhere(np.any(voxel, axis=3)))}')
#             # When using res = 1, there seems to be cost of accuracy, as there will be close points that become the same position
#             # print(f"{len(PDB_dict[Olfr]['coord'])}...{len(np.argwhere(np.any(voxel, axis=3)))}")

#          # Save the voxel to a file or do other processing as needed
#         voxel_list.append(voxel)
#     return voxel_list, voxel_shape, Olfr_order

# """
# Called by create_voxel. 
# When fill_radius=True. 
# Calls for assign_voxel, to fill in coordinates in voxels that are within the radius of the coordinate
# """
# def assign_voxel(voxel, voxel_index, num_points):
#     for i in range(-num_points, num_points + 1):
#         for j in range(-num_points, num_points + 1):
#             for k in range(-num_points, num_points + 1):
#                 distance = np.sqrt(i**2 + j**2 + k**2)
#                 if distance <= num_points:
#                     try:
#                         voxel[voxel_index[0] + i, voxel_index[1] + j, voxel_index[2] + k] = 1
#                     except IndexError:
#                         continue
# # OLD FUNCTIONING create_voxel ABOVE MODIFICATION ADDS RADIUS 
# # def create_voxel(coords, voxel_shape, resolution, spacer = [0,0,0]):
# #     # Initialize the voxel
# #     voxel = np.zeros(voxel_shape, dtype=int)
# #     # Compute the indices of the coordinates in the voxel
# #     indices = np.floor((coords - spacer) / resolution).astype(int)
# #     # Set the values of the voxel
# #     voxel[indices[:,0], indices[:,1], indices[:,2]] = 1
# #     return voxel


# """
# Get top features from a list of flattened_voxel_values 
# """
# def get_top_features(features, num_features = 10):

#     FEATURE_LIST = []
#     for i in features: 
#         indexed_feature = list(enumerate(i))
#         sorted_feature = sorted(indexed_feature, key=lambda x: x[1], reverse=True)
        
#         features = []
#         max_y = sorted_feature[0][1]
#         count = 0

#         for x, y in sorted_feature:
#             if y == max_y or count < num_features:
#                 features.append((x, y))
#                 count += 1
#             elif y < max_y:
#                 break
                
#         FEATURE_LIST.append(features)
    
#     return FEATURE_LIST

# """
# get_top_differnce function specifically takes in 2 lists of flatten voxels to conduct pairewise difference. 
# It simply finds the index with the maximum difference by substracting indices 
# """
# def get_top_difference(list_a, list_b, num_features=10):
    
#     if len(list_a) != len(list_b):
#         raise ValueError("Both lists must have the same length.")
        
#     # Calculate the absolute differences between corresponding elements
#     differences = [abs(a - b) for a, b in zip(list_a, list_b)]

#     # Create a list of tuples containing the differences and corresponding indices
#     indexed_differences = list(enumerate(differences))
#     # Sort the indexed differences based on the differences in descending order
#     sorted_differences = sorted(indexed_differences, key=lambda x: x[1], reverse=True)

#     # Get the top x differences and their indices
#     features = []
#     max_y = sorted_differences[0][1]
#     count = 0

#     for x, y in sorted_differences:
#         if y == max_y or count < num_features:
#             features.append((x, y))
#             count += 1
#         elif y < max_y:
#             break

#     return features

# """
# Get 3d voxel coordinate from flattened voxel indices. 

# cluster_voxel_data contains list of kClusters
# Within that list contains 4 np.arrays()
# [0,1,2] - voxel indice location 
# [4] - the percentage shared by Olfrs within the cluster 
# """
# def get_3Dcoord(features, pos_space, voxel_shape, max_scale=20, min_scale=10):

# # For testing
# # random_indice = np.array([random.randint(0, len(flat_voxel_data[0])) for _ in range(10000)])

#     FEATURE_3Dcoord = []
#     for feature in features:
#         indices, percent_shared = zip(*feature)

#         indices_3d = np.unravel_index(pos_space[list(indices)], voxel_shape) 
#         indices_3d = list(indices_3d)
#     #     Add percent_shared information. Scale between 10 and 5 for size plotting
#         indices_3d.append(np.array(scale(percent_shared, scale_between=[max_scale,
#                                                                         min_scale])))
#         FEATURE_3Dcoord.append(indices_3d)
#     return FEATURE_3Dcoord
        
# """
# Reverses the scale of a list of values such that the smallest value becomes 1 and the largest value becomes 0.
# """        
# def scale(values, reverse=False, factor = 1, scale_between = [1,0]):
#     min_val = min(values)
#     max_val = max(values)
#     new_max = scale_between[0]
#     new_min = scale_between[1]

#     if min_val != max_val: #if all the min and max is the same value. assign  max size 
#         scaled_values = [(value - min_val) * (new_max - new_min) / (max_val - min_val) + new_min for value in values]
#     else: 
#         scaled_values = [value*new_max for value in values]
# #     scaled_values = [(val - min_val) / (max_val - min_val)*factor for val in values]
#     if reverse:
#         scaled_values = [1 - val for val in scaled_values]
#     return scaled_values


# ATOM_RADIUS_DICT = {
#     'C': 1.70, 'CA': 1.80, 'CB': 1.90, 'CD': 1.88, 'CD1': 1.88, 'CD2': 1.88, 
#     'CE': 1.88, 'CE1': 1.88, 'CE2': 1.88, 'CE3': 1.88, 'CG': 1.88, 'CG1': 1.88 ,
#     'CG2': 1.88, 'CH2': 1.88, 'CZ': 1.88, 'CZ2': 1.88, 'CZ3': 1.88,
#     'N': 1.55, 'ND1': 1.55, 'ND2': 1.55, 'NE': 1.55, 'NE1': 1.55, 'NE2': 1.55,
#     'NH1': 1.55, 'NH2': 1.55, 'NZ': 1.55,
#     'O': 1.40, 'OD1': 1.40, 'OD2': 1.40, 'OE1': 1.40, 'OE2': 1.40, 'OG': 1.40 ,
#     'OG1': 1.40, 'OH': 1.40, 'OXT': 1.40, 'SD': 2.00, 'SG': 1.80}

# ATOM_ENCODING = {'C': [1, 0, 0, 0], 
#                  'N': [0, 1, 0, 0], 
#                  'O': [0, 0, 1, 0], 
#                  'S': [0, 0, 0, 1]}

AA_VDW_DICT = {
    'ALA': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB1': 1.487, '1HB': 1.487, 'HB2': 1.487, '2HB': 1.487, 'HB3': 1.487, '3HB': 1.487, 'C': 1.908, 'O': 1.6612}, 
    'ARG': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'HG2': 1.487, '2HG': 1.487, 'HG3': 1.487, 'HG1': 1.487, '1HG': 1.487, 'CD': 1.908, 'HD2': 1.387, '1HD': 1.387, '2HD': 1.387, 'HD3': 1.387, 'HD1': 1.387, 'NE': 1.75, 'HE': 0.6, 'CZ': 1.908, 'NH1': 1.75, 'HH11': 0.6, '1HH1': 0.6, 'HH12': 0.6, '2HH1': 0.6, 'NH2': 1.75, 'HH21': 0.6, '2HH2': 0.6, 'HH22': 0.6, '1HH2': 0.6, 'C': 1.908, 'O': 1.6612}, 
    'ASH': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, 'HB3': 1.487, 'CG': 1.908, 'OD1': 1.6612, 'OD2': 1.721, 'HD2': 0.0001, 'C': 1.908, 'O': 1.6612}, 
    'ASN': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'OD1': 1.6612, 'ND2': 1.824, 'HD21': 0.6, '1HD2': 0.6, 'HD22': 0.6, '2HD2': 0.6, 'C': 1.908, 'O': 1.6612}, 
    'ASP': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'OD1': 1.6612, 'OD2': 1.6612, 'C': 1.908, 'O': 1.6612}, 
    'CYM': {'N': 1.824, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB3': 1.387, 'HB2': 1.387, 'SG': 2.0, 'C': 1.908, 'O': 1.6612}, 
    'CYS': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.387, '2HB': 1.387, '1HB': 1.387, 'HB3': 1.387, 'HB1': 1.387, 'SG': 2.0, 'HG': 0.6, 'C': 1.908, 'O': 1.6612}, 
    'CYX': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.387, 'HB3': 1.387, 'SG': 2.0, 'C': 1.908, 'O': 1.6612}, 
    'GLH': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, 'HB3': 1.487, 'CG': 1.908, 'HG2': 1.487, 'HG3': 1.487, 'CD': 1.908, 'OE1': 1.6612, 'OE2': 1.721, 'HE2': 0.0001, 'C': 1.908, 'O': 1.6612}, 
    'GLN': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'HG2': 1.487, '2HG': 1.487, 'HG3': 1.487, 'HG1': 1.487, '1HG': 1.487, 'CD': 1.908, 'OE1': 1.6612, 'NE2': 1.824, 'HE21': 0.6, '1HE2': 0.6, 'HE22': 0.6, '2HE2': 0.6, 'C': 1.908, 'O': 1.6612}, 
    'GLU': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'HG2': 1.487, '2HG': 1.487, 'HG3': 1.487, 'HG1': 1.487, '1HG': 1.487, 'CD': 1.908, 'OE1': 1.6612, 'OE2': 1.6612, 'C': 1.908, 'O': 1.6612}, 
    'GLY': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA2': 1.387, 'HA1': 1.387, '1HA': 1.387, '2HA': 1.387, 'HA3': 1.387, 'C': 1.908, 'O': 1.6612}, 
    'HID': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, 'HB3': 1.487, 'CG': 1.85, 'ND1': 1.75, 'HD1': 0.6, 'CE1': 1.85, 'HE1': 1.359, 'NE2': 1.75, 'CD2': 2.0, 'HD2': 1.409, 'C': 1.908, 'O': 1.6612}, 
    'HIE': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, 'HB3': 1.487, 'CG': 1.85, 'ND1': 1.75, 'CE1': 1.85, 'HE1': 1.359, 'NE2': 1.75, 'HE2': 0.6, 'CD2': 2.0, 'HD2': 1.409, 'C': 1.908, 'O': 1.6612}, 
    'HIP': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, 'HB3': 1.487, 'CG': 1.85, 'ND1': 1.75, 'HD1': 0.6, 'CE1': 1.85, 'HE1': 1.359, 'NE2': 1.75, 'HE2': 0.6, 'CD2': 2.0, 'HD2': 1.409, 'C': 1.908, 'O': 1.6612}, 
    'ILE': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB': 1.487, 'CG2': 1.908, 'HG21': 1.487, '1HG2': 1.487, 'HG22': 1.487, '2HG2': 1.487, 'HG23': 1.487, '3HG2': 1.487, 'CG1': 1.908, 'HG12': 1.487, '2HG1': 1.487, 'HG13': 1.487, 'HG11': 1.487, '1HG1': 1.487, 'CD1': 1.908, 'HD11': 1.487, '1HD1': 1.487, 'HD12': 1.487, '2HD1': 1.487, 'HD13': 1.487, '3HD1': 1.487, 'C': 1.908, 'O': 1.6612}, 
    'LEU': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'HG': 1.487, 'CD1': 1.908, 'HD11': 1.487, '1HD1': 1.487, 'HD12': 1.487, '2HD1': 1.487, 'HD13': 1.487, '3HD1': 1.487, 'CD2': 1.908, 'HD21': 1.487, '1HD2': 1.487, 'HD22': 1.487, '2HD2': 1.487, 'HD23': 1.487, '3HD2': 1.487, 'C': 1.908, 'O': 1.6612}, 
    'LYN': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, 'HB3': 1.487, 'CG': 1.908, 'HG2': 1.487, 'HG3': 1.487, 'CD': 1.908, 'HD2': 1.487, 'HD3': 1.487, 'CE': 1.908, 'HE2': 1.1, 'HE3': 1.1, 'NZ': 1.824, 'HZ2': 0.6, 'HZ3': 0.6, 'C': 1.908, 'O': 1.6612}, 
    'LYS': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'HG2': 1.487, '2HG': 1.487, 'HG3': 1.487, 'HG1': 1.487, '1HG': 1.487, 'CD': 1.908, 'HD2': 1.487, '1HD': 1.487, '2HD': 1.487, 'HD3': 1.487, 'HD1': 1.487, 'CE': 1.908, 'HE2': 1.1, '2HE': 1.1, 'HE3': 1.1, '1HE': 1.1, 'HE1': 1.1, 'NZ': 1.824, 'HZ1': 0.6, '1HZ': 0.6, 'HZ2': 0.6, '2HZ': 0.6, 'HZ3': 0.6, '3HZ': 0.6, 'C': 1.908, 'O': 1.6612}, 
    'MET': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'HG2': 1.387, '2HG': 1.387, 'HG3': 1.387, 'HG1': 1.387, '1HG': 1.387, 'SD': 2.0, 'CE': 1.908, 'HE1': 1.387, '1HE': 1.387, 'HE2': 1.387, '2HE': 1.387, 'HE3': 1.387, '3HE': 1.387, 'C': 1.908, 'O': 1.6612}, 
    'PHE': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'CD1': 1.908, 'HD1': 1.459, 'CE1': 1.908, 'HE1': 1.459, 'CZ': 1.908, 'HZ': 1.459, 'CE2': 1.908, 'HE2': 1.459, 'CD2': 1.908, 'HD2': 1.459, 'C': 1.908, 'O': 1.6612}, 
    'PRO': {'N': 1.824, 'CD': 1.908, 'HD2': 1.387, '1HD': 1.387, '2HD': 1.387, 'HD3': 1.387, 'HD1': 1.387, 'CG': 1.908, 'HG2': 1.487, '2HG': 1.487, 'HG3': 1.487, 'HG1': 1.487, '1HG': 1.487, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CA': 1.908, 'HA': 1.387, 'C': 1.908, 'O': 1.6612}, 
    'SER': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.387, '2HB': 1.387, '1HB': 1.387, 'HB3': 1.387, 'HB1': 1.387, 'OG': 1.721, 'HG': 0.0001, 'C': 1.908, 'O': 1.6612}, 
    'THR': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB': 1.387, 'CG2': 1.908, 'HG21': 1.487, '1HG2': 1.487, 'HG22': 1.487, '2HG2': 1.487, 'HG23': 1.487, '3HG2': 1.487, 'OG1': 1.721, 'HG1': 0.0001, 'C': 1.908, 'O': 1.6612}, 
    'TRP': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.85, 'CD1': 2.0, 'HD1': 1.409, 'NE1': 1.75, 'HE1': 0.6, 'CE2': 1.85, 'CZ2': 1.908, 'HZ2': 1.459, 'CH2': 1.908, 'HH2': 1.459, 'CZ3': 1.908, 'HZ3': 1.459, 'CE3': 1.908, 'HE3': 1.459, 'CD2': 1.85, 'C': 1.908, 'O': 1.6612}, 
    'TYR': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.908, 'CD1': 1.908, 'HD1': 1.459, 'CE1': 1.908, 'HE1': 1.459, 'CZ': 1.908, 'OH': 1.721, 'HH': 0.0001, 'CE2': 1.908, 'HE2': 1.459, 'CD2': 1.908, 'HD2': 1.459, 'C': 1.908, 'O': 1.6612}, 
    'VAL': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB': 1.487, 'CG1': 1.908, 'CG2': 1.908, 'HG11': 1.487, '1HG2': 1.487, '1HG1': 1.487, 'HG21': 1.487, 'HG12': 1.487, '2HG1': 1.487, 'HG22': 1.487, '2HG2': 1.487, 'HG13': 1.487, '3HG2': 1.487, '3HG1': 1.487, 'HG23': 1.487, 'C': 1.908, 'O': 1.6612}, 
    'HIS': {'N': 1.824, 'H': 0.6, 'HN': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, '2HB': 1.487, '1HB': 1.487, 'HB3': 1.487, 'HB1': 1.487, 'CG': 1.85, 'ND1': 1.75, 'HD1': 0.6, 'CE1': 1.85, 'HE1': 1.359, 'NE2': 1.75, 'CD2': 2.0, 'HD2': 1.409, 'C': 1.908, 'O': 1.6612}, 
    'PTR': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, 'HB3': 1.487, 'CG': 1.908, 'CD1': 1.908, 'HD1': 1.459, 'CE1': 1.908, 'HE1': 1.459, 'CZ': 1.908, 'CE2': 1.908, 'HE2': 1.459, 'CD2': 1.908, 'HD2': 1.459, 'OH': 1.6837, 'P': 2.1, 'O1P': 1.85, 'O2P': 1.85, 'O3P': 1.85, 'C': 1.908, 'O': 1.6612}, 
    'SEP': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.387, 'HB3': 1.387, '1HB': 1.387, '2HB': 1.387, 'OG': 1.6837, 'P': 2.1, 'O1P': 1.85, 'O2P': 1.85, 'O3P': 1.85, 'C': 1.908, 'O': 1.6612}, 
    'TPO': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB': 1.387, 'CG2': 1.908, 'HG21': 1.487, 'HG22': 1.487, 'HG23': 1.487, '1HG2': 1.487, '2HG2': 1.487, '3HG2': 1.487, 'OG1': 1.6837, 'P': 2.1, 'O1P': 1.85, 'O2P': 1.85, 'O3P': 1.85, 'C': 1.908, 'O': 1.6612}, 
    'H2D': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, 'HB3': 1.487, 'CG': 1.85, 'ND1': 1.75, 'CE1': 1.85, 'HE1': 1.359, 'NE2': 1.75, 'HE2': 0.6, 'CD2': 2.0, 'HD2': 1.409, 'P': 2.1, 'O1P': 1.85, 'O2P': 1.85, 'O3P': 1.85, 'C': 1.908, 'O': 1.6612}, 
    'Y1P': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.487, 'HB3': 1.487, 'CG': 1.908, 'CD1': 1.908, 'HD1': 1.459, 'CE1': 1.908, 'HE1': 1.459, 'CZ': 1.908, 'CE2': 1.908, 'HE2': 1.459, 'CD2': 1.908, 'HD2': 1.459, 'OG': 1.6837, 'P': 2.1, 'O1P': 1.721, 'O2P': 1.6612, 'O3P': 1.6612, 'H1P': 0.0001, 'C': 1.908, 'O': 1.6612}, 
    'T1P': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB': 1.387, 'CG2': 1.908, 'HG21': 1.487, 'HG22': 1.487, 'HG23': 1.487, 'OG': 1.6837, 'P': 2.1, 'O1P': 1.721, 'O2P': 1.6612, 'O3P': 1.6612, 'H1P': 0.0001, 'C': 1.908, 'O': 1.6612}, 
    'S1P': {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.387, 'CB': 1.908, 'HB2': 1.387, 'HB3': 1.387, 'OG': 1.6837, 'P': 2.1, 'O1P': 1.721, 'O2P': 1.6612, 'O3P': 1.6612, 'H1P': 0.0001, 'C': 1.908, 'O': 1.6612}, 
    'GEN': {'AC': 2.0, 'AG': 1.72, 'AL': 2.0, 'AM': 2.0, 'AR': 1.88, 'AS': 1.85, 'AT': 2.0, 'AU': 1.66, 'B': 2.0, 'BA': 2.0, 'BE': 2.0, 'BH': 2.0, 'BI': 2.0, 'BK': 2.0, 'BR': 1.85, 'C': 1.66, 'CA': 2.0, 'CD': 1.58, 'CE': 2.0, 'CF': 2.0, 'CL': 1.75, 'CM': 2.0, 'CO': 2.0, 'CR': 2.0, 'CS': 2.0, 'CU': 1.4, 'DB': 2.0, 'DS': 2.0, 'DY': 2.0, 'ER': 2.0, 'ES': 2.0, 'EU': 2.0, 'F': 1.47, 'FE': 2.0, 'FM': 2.0, 'FR': 2.0, 'GA': 1.87, 'GD': 2.0, 'GE': 2.0, 'H': 0.91, 'HE': 1.4, 'HF': 2.0, 'HG': 1.55, 'HO': 2.0, 'HS': 2.0, 'I': 1.98, 'IN': 1.93, 'IR': 2.0, 'K': 2.75, 'KR': 2.02, 'LA': 2.0, 'LI': 1.82, 'LR': 2.0, 'LU': 2.0, 'MD': 2.0, 'MG': 1.73, 'MN': 2.0, 'MO': 2.0, 'MT': 2.0, 'N': 1.97, 'NA': 2.27, 'NB': 2.0, 'ND': 2.0, 'NE': 1.54, 'NI': 1.63, 'NO': 2.0, 'NP': 2.0, 'O': 1.69, 'OS': 2.0, 'P': 2.1, 'PA': 2.0, 'PB': 2.02, 'PD': 1.63, 'PM': 2.0, 'PO': 2.0, 'PR': 2.0, 'PT': 1.72, 'PU': 2.0, 'RA': 2.0, 'RB': 2.0, 'RE': 2.0, 'RF': 2.0, 'RH': 2.0, 'RN': 2.0, 'RU': 2.0, 'S': 2.09, 'SB': 2.0, 'SC': 2.0, 'SE': 1.9, 'SG': 2.0, 'SI': 2.1, 'SM': 2.0, 'SN': 2.17, 'SR': 2.0, 'TA': 2.0, 'TB': 2.0, 'TC': 2.0, 'TE': 2.06, 'TH': 2.0, 'TI': 2.0, 'TL': 1.96, 'TM': 2.0, 'U': 1.86, 'V': 2.0, 'W': 2.0, 'XE': 2.16, 'Y': 2.0, 'YB': 2.0, 'ZN': 1.39, 'ZR': 2.0}
               }

AA_RES_OHE = {
    'ALA':  1,'GLY':  2,'ILE':  3,'LEU':  4,'MET':  5,
    'VAL':  6,'PHE':  7,'TRP':  8,'TYR':  9,'ASN': 10,
    'CYS': 11,'GLN': 12,'PRO': 13,'SER': 14,'THR': 15, 
    'ASP': 16,'GLU': 17,'ARG': 18,'HIS': 19,'LYS': 20,                             
    'UNK': 21
}

AA_OHE_BASIC = {
    'ALA': 1,'GLY': 1,'ILE': 1,'LEU': 1,'MET': 1,'VAL': 1,  # Alipathic apolar
    'PHE': 2,'TRP': 2,'TYR': 2,                             # Aromatic 
    'ASN': 3,'CYS': 3,'GLN': 3,'PRO': 3,'SER': 3,'THR': 3,  # Polar uncharged
    'ASP': 4,'GLU': 4,                                      # Negatively charged 
    'ARG': 5,'HIS': 5,'LYS': 5,                             # Positively charged
    'UNK': 6,'NON': 6                                       # Non-standard
}
# Categorize amino acids by key binding-relevant properties
AA_OHE = {
    # Hydrophobicity (crucial for binding interactions)
    'hydrophobicity': {
        'very_hydrophobic': ['LEU', 'ILE', 'VAL', 'MET', 'PHE', 'TRP'],
        'hydrophobic': ['ALA', 'PRO', 'CYS'],
        'neutral': ['THR', 'SER', 'GLY'],
        'hydrophilic': ['ASP', 'GLU', 'LYS', 'ARG', 'HIS']
    },
    
    # Charge (important for electrostatic interactions)
    'charge': {
        'positive': ['LYS', 'ARG', 'HIS'],
        'negative': ['ASP', 'GLU'],
        'neutral': ['ALL_OTHER']
    },
    
    # H-bond capability (critical for specific interactions)
    'h_bond': {
        'donor': ['SER', 'THR', 'TYR', 'LYS', 'ARG', 'HIS', 'TRP', 'ASN', 'GLN'],
        'acceptor': ['ASP', 'GLU', 'ASN', 'GLN'],
        'both': ['SER', 'THR', 'TYR'],
        'none': ['ALA', 'LEU', 'ILE', 'VAL', 'PRO', 'PHE', 'MET']
    },
    
    # Aromaticity (important for π-π and cation-π interactions)
    'aromaticity': {
        'aromatic': ['PHE', 'TYR', 'TRP', 'HIS'],
        'non_aromatic': ['ALL_OTHER']
    },
    
    # Size and flexibility (affects binding pocket interactions)
    'size_flexibility': {
        'small_rigid': ['ALA', 'GLY', 'PRO'],
        'small_flexible': ['SER', 'THR', 'CYS'],
        'medium_flexible': ['ASN', 'GLN', 'ASP', 'GLU'],
        'large_flexible': ['LYS', 'ARG', 'MET'],
        'large_hydrophobic': ['LEU', 'ILE', 'VAL', 'PHE', 'TRP']
    }
}

# Reverse mapping for efficient lookups
def create_reverse_property_mapping(property_dict):
    """
    Create a reverse mapping for efficient property lookups
    """
    reverse_map = {}
    for prop, amino_acids in property_dict.items():
        for aa_list in amino_acids.values():
            for aa in aa_list:
                if aa != 'ALL_OTHER':
                    if aa not in reverse_map:
                        reverse_map[aa] = {}
                    reverse_map[aa][prop] = list(amino_acids.keys())[list(amino_acids.values()).index(aa_list)]
    return reverse_map

# Create lookup dictionary
AA_PROPERTY_LOOKUP = create_reverse_property_mapping(AA_OHE)

def one_hot_encode_aa_properties(residue):
    """
    One-hot encode amino acid properties relevant to ligand binding
    
    Parameters:
    -----------
    residue : str
        Three-letter amino acid code
    
    Returns:
    --------
    list
        One-hot encoded vector of binding-relevant properties
    """
    # Default to all zeros if residue not found
    encoding = np.zeros(20, dtype=int)
    
    # If residue not in lookup, return zero vector
    if residue not in AA_PROPERTY_LOOKUP:
        return encoding.tolist()
    
    # Get properties for this residue
    props = AA_PROPERTY_LOOKUP[residue]
    
    # Encode hydrophobicity
    hydro_map = {'very_hydrophobic': 0, 'hydrophobic': 1, 'neutral': 2, 'hydrophilic': 3}
    encoding[hydro_map.get(props.get('hydrophobicity', 'neutral'), 2)] = 1
    
    # Encode charge
    charge_map = {'positive': 4, 'negative': 5, 'neutral': 6}
    encoding[charge_map.get(props.get('charge', 'neutral'), 6)] = 1
    
    # Encode H-bond capability
    h_bond_map = {'donor': 7, 'acceptor': 8, 'both': 9, 'none': 10}
    encoding[h_bond_map.get(props.get('h_bond', 'none'), 10)] = 1
    
    # Encode aromaticity
    arom_map = {'aromatic': 11, 'non_aromatic': 12}
    encoding[arom_map.get(props.get('aromaticity', 'non_aromatic'), 12)] = 1
    
    # Encode size and flexibility
    size_flex_map = {
        'small_rigid': 13, 'small_flexible': 14, 
        'medium_flexible': 15, 'large_flexible': 16, 
        'large_hydrophobic': 17
    }
    encoding[size_flex_map.get(props.get('size_flexibility', 'small_rigid'), 13)] = 1
    
    # Add two additional features for special cases
    encoding[18] = 1 if residue in ['CYS'] else 0  # Special sulfur-containing
    encoding[19] = 1 if residue in ['PRO'] else 0  # Unique cyclic structure
    
    return encoding.tolist()


def encode_voxels(res_data, mode='ohe', OR_esm=None, esm_order=None):
    """
    Encodes residue coordinates into feature vectors suitable for voxelization.

    This function converts 3D residue coordinates into an enriched representation
    by attaching feature vectors that describe residue identity or properties.
    It supports three main encoding schemes:

    1. `'ohe'`         : One-hot encoding of all 20 amino acids + 1 for unknown
    2. `'ohe_basic'`   : One-hot encoding of grouped amino acid classes
    3. `'esm'`         : Encodes each residue using its corresponding ESM embedding

    Parameters
    ----------
    residue_coords : list of list
        Each entry corresponds to a list of residue coordinate entries for a receptor.
        Each residue entry should be a list or tuple:
            [x, y, z, residue_name, radius (optional)]

        For example:
            [
                [[12.0, 23.5, 45.3, 'A', 1.8], [13.1, 24.2, 44.7, 'C', 1.7], ...],
                ...
            ]
        If radius is not provided, a default value of 1.5 Å is used.

    mode : str, default='ohe'
        The encoding scheme to use for residues:
        - `'ohe'`       : Standard one-hot encoding for 20 amino acids + 1 unknown
        - `'ohe_basic'` : Basic classification (nonpolar, polar, acidic, etc.)
        - `'esm'`       : Embeds residues using ESM vectors (requires OR_esm & esm_order)

    OR_esm : dict, optional
        Required for `'esm'` mode. Maps OR names to their full ESM-1b embeddings.
        Should be structured as:
            {
                'OR1A1': np.ndarray of shape (L, 1280),
                'OR2T6': ...,
                ...
            }

    esm_order : list of str, optional
        Required for `'esm'` mode. List of OR names corresponding to `residue_coords`
        order so that ESM embeddings align with their residues.

    Returns
    -------
    list of list of list
        Returns a nested list structure of encoded coordinates:
        [
            [[x, y, z, feature_vector, radius], [...], ...],  # for first OR
            [[x, y, z, feature_vector, radius], [...], ...],  # for second OR
            ...
        ]

    Notes
    -----
    - In `'esm'` mode, the feature vector comes directly from the ESM embedding for
      each residue (1280-dimensional by default).
    - In `'ohe'` and `'ohe_basic'` modes, the feature vector is a shorter one-hot vector
      based on amino acid identity or class.
    - This function standardizes encoding for downstream voxelization or ML applications.
    - If any residue name is unrecognized, it is mapped to a default "unknown" category.

    Examples
    --------
    >>> encode_voxels(residue_coords, mode='ohe')
    >>> encode_voxels(residue_coords, mode='esm', OR_esm=esm_dict, esm_order=OR_labels)
    """

    translated_data = []

    for i, _olfr in enumerate(res_data):
        _olfr_res = []
        # Only needed for esm
        if mode == 'esm': 
            if len(OR_esm) and len(esm_order):
                _or_name = esm_order[i].split('_')[0]
                esm = OR_esm[_or_name] if mode == 'esm' else None
                # esm = OR_esm[_olfr.split('_')[0]]
                # print(_or_name)
            else:
                raise ValueError("OR_esm and esm_order must be provided for 'esm' mode.")

        for _res in _olfr:
            x, y, z = map(float, _res[3:6])
            residue_number = int(_res[0])
            residue = _res[1]
            atom_name = _res[2]
            element = ''.join(filter(str.isalpha, atom_name.upper()))
            vdw = AA_VDW_DICT.get(residue, {}).get(element, 1)

            # Feature vector logic
            if mode == 'ohe':
                feature_vector = torch.tensor([0] + one_hot_encode_aa_properties(residue), dtype=torch.float32)
            elif mode == 'ohe_basic':
                cls = AA_OHE_BASIC.get(residue, 6)
                feature_vector = torch.zeros(len(set(AA_OHE_BASIC.values())) + 1, dtype=torch.float32)
                feature_vector[cls] = 1
            elif mode == 'res': 
                feature_vector = torch.tensor([residue_number, AA_RES_OHE.get(residue)], dtype=torch.float32)
            elif mode == 'esm':
                #TODO This is a hot fix . . .
                if len(esm) <= residue_number-1: 
                    continue
                feature_vector = torch.tensor([0] + list(esm[residue_number - 1]), dtype=torch.float32)
            else:
                raise ValueError(f"Unknown encoding mode: {mode}")

            _olfr_res.append([x, y, z, feature_vector, vdw])
        translated_data.append(_olfr_res)

    return translated_data

def voxelize_cavity(
    cavity_coords=None, 
    residue_coords=None, 
    OR_esm=None, 
    esm_order=None, 
    resolution=1, 
    encode_method='ohe', 
    vdw_radius=True, 
    sparse_mode=False,
    reference_center=None,
    cube_dim=32
):
    """
    Voxelizes 3D coordinates by creating a voxel grid representation.

    Parameters
    ----------
    cavity_coords : dict or list, optional
        3D coordinates for cavities.
    residue_coords : dict or list, optional
        Residue interaction data.
    OR_esm : dict, optional
        Dictionary mapping residue identifiers to ESM embeddings.
    esm_order : list, optional
        Order of residues for ESM alignment.
    resolution : float, default 1
        Size of each voxel (angstrom).
    encode_method : str, default 'ohe'
        Method of encoding residue properties ('ohe', 'esm', etc.).
    vdw_radius : bool, default True
        Whether to expand each coordinate based on its van der Waals radius.
    sparse_mode : bool, default False
        If True, uses sparse voxel pooling (memory-efficient, no zero-padding).
    reference_center : np.ndarray, shape (3,), optional
        If provided, fixes the voxel cube center for all inputs.
    cube_dim : float, optional
        Edge length of the voxel cube (in Å). Required if reference_center is given.

    Returns
    -------
    tuple: 
        - List of voxel grids (or sparse dicts if sparse_mode=True)
        - Tuple of grid shape (X, Y, Z)
    """
    
    from collections import defaultdict
    from collections import Counter

    # Convert dict to list if necessary
    cavity_coords = list(cavity_coords.values()) if isinstance(cavity_coords, dict) else (cavity_coords or [])
    residue_coords = list(residue_coords.values()) if isinstance(residue_coords, dict) else (residue_coords or [])

    # Prepare residue coordinates based on encoding method     
    print(f'Encoding voxels via {encode_method}...')
    residue_coords_class = encode_voxels(
        residue_coords,
        mode=encode_method, 
        OR_esm=OR_esm, 
        esm_order=esm_order
    ) if residue_coords else None

    # Collect all coordinates
    all_coords = []
    if cavity_coords:
        all_coords.extend(np.concatenate(cavity_coords, axis=0))
    if residue_coords_class:
        all_coords.extend(
            np.array([_coord[:3] for _all_coord in residue_coords_class for _coord in _all_coord])
        )

    # Handle case with no coordinates
    if not all_coords:
        return [], (0, 0, 0)

    # Convert to numpy array for processing
    all_coords = np.array(all_coords)
    
    # Fixed cube mode - Use a reference point for reproducibility 
    if reference_center is not None and cube_dim is not None:
        half = cube_dim / 2.0
        min_coords = reference_center - half
        max_coords = reference_center + half
        grid_shape = np.ceil((max_coords - min_coords) / resolution).astype(int)
    else:
        # Default: per-OR dynamic cube
        min_coords = np.min(all_coords, axis=0)
        max_coords = np.max(all_coords, axis=0)
        grid_shape = np.ceil((max_coords - min_coords) / resolution).astype(int) + 1

    # Determine vector length dynamically
    vector_length = len(residue_coords_class[0][0][3])

    # Initialize voxelized data
    voxelized_data = []

    # Process cavities and/or residue coordinates
    num_iterations = max(
        len(cavity_coords) if cavity_coords else 0, 
        len(residue_coords_class) if residue_coords_class else 0
    )
    
    print(f'Generating voxels. . .') 
    for i in tqdm(range(num_iterations)):
        
        if sparse_mode:
            # Use sparse voxel pooling
            voxel_dict = defaultdict(list)

            if residue_coords_class and i < len(residue_coords_class):            
                for point in residue_coords_class[i]:
                    x, y, z = point[:3]
                    features = point[3]
                    radius = point[4] if vdw_radius else 0

                    grid_x = int((x - min_coords[0]) // resolution)
                    grid_y = int((y - min_coords[1]) // resolution)
                    grid_z = int((z - min_coords[2]) // resolution)

                    r_voxels = int(math.ceil(radius / resolution)) if vdw_radius else 0

                    for dx in range(-r_voxels, r_voxels + 1):
                        for dy in range(-r_voxels, r_voxels + 1):
                            for dz in range(-r_voxels, r_voxels + 1):
                                dist = math.sqrt((dx * resolution) ** 2 + 
                                                 (dy * resolution) ** 2 + 
                                                 (dz * resolution) ** 2)
                                if dist <= radius:
                                    nx, ny, nz = grid_x + dx, grid_y + dy, grid_z + dz
                                    voxel_dict[(nx, ny, nz)].append(features)

                    if not vdw_radius:
                        voxel_dict[(grid_x, grid_y, grid_z)].append(features)

            # pooled_voxels = {}
            # for coord, feats in voxel_dict.items():
            #     pooled_voxels[coord] = torch.stack(feats).mean(dim=0)
            # Majority voting pooling instead of averaging pooling
            pooled_voxels = {}
            for coord, feats in voxel_dict.items():
                if encode_method == 'res':
                    # Cast all feature vectors to integer tuples (res_num, res_idx)
                    tuples = []
                    for f in feats:
                        res_num = int(round(float(f[0].item())))
                        res_idx = int(round(float(f[1].item())))
                        tuples.append((res_num, res_idx))

                    # Majority vote across residues in this voxel
                    most_common, _ = Counter(tuples).most_common(1)[0]
                    pooled_voxels[coord] = torch.tensor(
                        [float(most_common[0]), float(most_common[1])],
                        dtype=torch.float32
                    )
                else:
                    # For continuous encodings (ohe, esm), average still makes sense
                    pooled_voxels[coord] = torch.stack(feats).mean(dim=0)

            voxelized_data.append(pooled_voxels)
            
        else:
            # Use dense voxel grid
            voxel_grid = torch.zeros((*grid_shape, vector_length), dtype=torch.float32)
            
            # Add residue coordinates if available
            if residue_coords_class and i < len(residue_coords_class):            
                for point in residue_coords_class[i]:
                    x, y, z = point[:3]
                    features = point[3]

                    grid_x = int((x - min_coords[0]) // resolution)
                    grid_y = int((y - min_coords[1]) // resolution)
                    grid_z = int((z - min_coords[2]) // resolution)

                    # expand voxel coverage around the point using the radius
                    if vdw_radius:
                        radius = point[4]
                        r_voxels = int(math.ceil(radius / resolution))
                        for dx in range(-r_voxels, r_voxels + 1):
                            for dy in range(-r_voxels, r_voxels + 1):
                                for dz in range(-r_voxels, r_voxels + 1):
                                    dist = math.sqrt((dx * resolution) ** 2 + 
                                                    (dy * resolution) ** 2 + 
                                                    (dz * resolution) ** 2)
                                    if dist <= radius:
                                        nx, ny, nz = grid_x + dx, grid_y + dy, grid_z + dz
                                        if 0 <= nx < voxel_grid.shape[0] and \
                                        0 <= ny < voxel_grid.shape[1] and \
                                        0 <= nz < voxel_grid.shape[2]:
                                            voxel_grid[nx, ny, nz] += features
                                            
                    else: 
                        voxel_grid[grid_x, grid_y, grid_z] = features
            
            # Add cavity coordinates if available
            coords_data = []
            if cavity_coords and i < len(cavity_coords):
                # First feature is cavity presence, then populates the feature manually as 0
                cavity_flag_vector = torch.zeros(vector_length, dtype=torch.float32)
                cavity_flag_vector[0] = 1.0
                
                for _coord in cavity_coords[i]:
                    x, y, z = _coord[:3]
                    grid_x = int((x - min_coords[0]) // resolution)
                    grid_y = int((y - min_coords[1]) // resolution)
                    grid_z = int((z - min_coords[2]) // resolution)
                    
                    # Clamp to valid range, Ensures doesn't index out of grid.
                    grid_x = max(0, min(grid_x, voxel_grid.shape[0] - 1))
                    grid_y = max(0, min(grid_y, voxel_grid.shape[1] - 1))
                    grid_z = max(0, min(grid_z, voxel_grid.shape[2] - 1))
                    # voxel_grid[grid_x, grid_y, grid_z] = cavity_flag_vector 
                    # Only set the flag, preserving existing residue features
                    # print(voxel_grid.shape, grid_x, grid_y, grid_z)
                    voxel_grid[grid_x, grid_y, grid_z][0] = 1.0
                
            voxelized_data.append(voxel_grid)

    return voxelized_data, grid_shape

def sparse_avg_pool3d(sparse_voxel, grid_shape, kernel_size=4, stride=4):
    """
    Pools sparse voxels using average pooling over 3D windows.

    Parameters:
    -----------
    sparse_voxel : dict
        Keys: (x, y, z) voxel indices  
        Values: torch.Tensor (feature vector)
    grid_shape : tuple
        The (X, Y, Z) shape of the grid to define pooling bounds.
    kernel_size : int
        Size of the 3D window.
    stride : int
        Step size between pooled windows.

    Returns:
    --------
    pooled_features : torch.Tensor
        Pooled feature vectors per 3D window. Shape: (N_blocks, feature_dim)
    """
    pooled = []
    feature_dim = next(iter(sparse_voxel.values())).shape[0]

    X, Y, Z = grid_shape
    for x in range(0, X, stride):
        for y in range(0, Y, stride):
            for z in range(0, Z, stride):
                window_feats = []
                for dx in range(kernel_size):
                    for dy in range(kernel_size):
                        for dz in range(kernel_size):
                            voxel_coord = (x + dx, y + dy, z + dz)
                            if voxel_coord in sparse_voxel:
                                window_feats.append(sparse_voxel[voxel_coord])
                if window_feats:
                    pooled.append(torch.stack(window_feats).mean(dim=0))
                else:
                    pooled.append(torch.zeros(feature_dim))  # pad with zero if empty
    return torch.stack(pooled)  # (N_blocks, feature_dim)

# def voxelize_coordinates(cavity_coords, resolution=1):
#     """
#     ********** DEPRECATED ********** 
#     USE voxelize_cavity INSTEAD. 
    
#     Voxelizes the 3D coordinates by placing 1s in voxels that are occupied by coordinates.
    
#     :param cavity_coords: List of lists containing 3D coordinates for each Olfr
#     :param resolution: Size of each voxel (default is 1)
#     :return: List of 1D arrays representing the voxelized space for each Olfr
#     """
#     # Step 1: Find the global min and max coordinates across all cavities
#     all_coords = np.concatenate(cavity_coords, axis=0)
#     min_coords = np.min(all_coords, axis=0)
#     max_coords = np.max(all_coords, axis=0)
    
#     # Step 2: Define voxel grid shape
#     grid_shape = np.ceil((max_coords - min_coords) / resolution).astype(int)
    
#     # Step 3: Create a 3D grid for each cavity
#     voxelized_data = []
    
#     for cavity in cavity_coords:
#         # Step 4: Create an empty voxel grid
#         voxel_grid = np.zeros(grid_shape, dtype=int)
        
#         # Step 5: Convert each cavity point to voxel grid indices
#         for point in cavity:
#             # Translate point into grid coordinates
#             grid_x = int((point[0] - min_coords[0]) // resolution)
#             grid_y = int((point[1] - min_coords[1]) // resolution)
#             grid_z = int((point[2] - min_coords[2]) // resolution)
            
#             # Mark the voxel as occupied
#             voxel_grid[grid_x, grid_y, grid_z] = 1
        
#         voxelized_data.append(voxel_grid)
    
#     return voxelized_data, grid_shape


# Convert OHE properties into single integer labels (0-6)
def convert_properties(voxel_array):
    """
    Converts one-hot encoded (OHE) residue properties into a single integer label for each voxel.

    This function translates a voxel grid with one-hot encoded residue classes into a simplified 
    integer-labeled format. The first index (0) indicates cavity presence, while residue classifications 
    start at index 1.

    Parameters:
    -----------
    voxel_array : numpy.ndarray
        A 4D numpy array (X, Y, Z, 7), where the last dimension contains one-hot encoded classifications.

    Returns:
    --------
    numpy.ndarray
        A 3D numpy array (X, Y, Z) where each voxel is labeled with an integer (0-6):
        - 0 → Empty space
        - 1 → Aliphatic apolar residues
        - 2 → Aromatic residues
        - 3 → Polar uncharged residues
        - 4 → Negatively charged residues
        - 5 → Positively charged residues
        - 6 → Non-standard residues

    Example:
    --------
    Input:
    voxel_array[x, y, z] = [0, 0, 0, 1, 0, 0, 0]

    Output:
    labeled_voxel[x, y, z] = 3
    """
    
    labeled_voxel = np.zeros(voxel_array.shape[:3], dtype=int)  # Initialize empty grid

    for x in range(voxel_array.shape[0]):
        for y in range(voxel_array.shape[1]):
            for z in range(voxel_array.shape[2]):
                properties = voxel_array[x, y, z]  # 7-length OHE
                if np.any(properties):  
                    labeled_voxel[x, y, z] = np.argmax(properties)  # Directly use the index
                else:
                    labeled_voxel[x, y, z] = -1  # -1 used for empty space (as np.argmax[1,0,0 ...] also denotes to 0)

    return labeled_voxel

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def coords_to_voxel_pca(cavity_coords=None, residue_coords=None, resolution=1, n_components=2):
    """
    Processes voxelized cavity data and applies PCA to reduce dimensions.

    Parameters:
    -----------
    cavity_coords : dict
        Dictionary of cavity coordinates.
    residue_coords : dict
        Dictionary of residue coordinates.
    resolution : int, optional
        Resolution for voxelization (default is 1).
    n_components : int, optional
        Number of PCA components to retain (default is 2).

    Returns:
    --------
    pca_df : pd.DataFrame
        DataFrame containing PCA-transformed coordinates with OR labels.
    variance_ratio : np.ndarray
        Explained variance ratio of each principal component.
    """
    
    
    
    # Voxelize the cavity
    voxelized_array, voxel_shape = voxelize_cavity(
        cavity_coords=list(cavity_coords.values()) if cavity_coords is not None else cavity_coords,
        residue_coords=list(residue_coords.values())if residue_coords is not None else residue_coords,
        resolution=resolution
    )

    # Convert voxel properties
    labeled_voxels = np.array([convert_properties(voxel) for voxel in voxelized_array])

    # Flatten voxel grids
    flattened_voxels = np.array([voxel.flatten() for voxel in labeled_voxels])  # Shape (num_ORs, num_voxels)

    # Remove constant columns
    non_constant_mask = np.any(flattened_voxels != flattened_voxels[0, :], axis=0)
    filtered_voxels = flattened_voxels[:, non_constant_mask]  # Shape (num_ORs, reduced_voxel_features)

    print(f"Original features: {flattened_voxels.shape[1]}, Reduced features: {filtered_voxels.shape[1]}")

    # Standardize for PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_voxels)

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)
    variance_ratio = pca.explained_variance_ratio_

    # Store results in a DataFrame
    pca_df = pd.DataFrame(reduced_data, columns=[f'PCA_{i+1}' for i in range(n_components)], index=list(residue_coords.keys()))
    pca_df = pca_df.reset_index().rename(columns={'index': 'or_cid'})

    print("Reduced data shape:", reduced_data.shape)
    print("Explained variance ratio:", variance_ratio)

    return pca_df, variance_ratio

from collections import defaultdict, Counter
def sparse_voxel_consensus_res(voxel_dicts, threshold=0.0, use_bw=False):
    """
    Build a consensus residue mapping across multiple sparse voxel dictionaries.

    Each voxel dictionary should come from voxelization using 
    `vf.voxelize_cavity(..., encode_method='res', sparse_mode=True, ...)`, 
    where keys are voxel coordinates (x, y, z) and values are either:
        - tensor([residue_number, residue_index])         if use_bw=False
        - (residue_number, residue_index, bw_number)      if use_bw=True

    Parameters
    ----------
    voxel_dicts : list of dict
        A list of sparse voxel dictionaries, typically generated via:
        voxels, grid_shape = vf.voxelize_cavity(
            cavity_coords=Cbc_cav_coords,
            residue_coords=Cbc_res_coords,
            resolution=1,
            cube_dim=32,
            reference_center=center_reference,
            encode_method='res',
            sparse_mode=True,
            vdw_radius=False
        )

    threshold : float, optional (default=0.0)
        Minimum fraction of input voxel dictionaries in which the consensus 
        residue must appear at a given voxel coordinate for it to be included. 
        For example, with `threshold=0.1`, the most frequent residue must 
        occur in at least 10% of all voxel dictionaries.

    use_bw : bool, optional (default=False)
        If True, consensus is determined using residue index **and** BW number. 
        The consensus result will be a tuple `(res_idx, bw_number)`.

    Returns
    -------
    dict
        A consensus mapping of voxel coordinates:
        - If use_bw=False → {coord: (res_idx, freq_fraction)}
        - If use_bw=True  → {coord: (res_idx, bw_number, freq_fraction)}

        Voxels that do not meet the threshold are excluded.

    Notes
    -----
    - Counts are summarized as frequency fraction for interpretability.
    - Using `use_bw=True` is useful for directly labeling voxels with 
      Ballesteros–Weinstein (BW) numbers in downstream visualization.
    """
    voxel_res_counts = defaultdict(list)

    # Collect residue IDs (and BW if requested)
    for d in voxel_dicts:
        for coord, val in d.items():
            if use_bw:
                # expecting tuple (res_num, res_idx, bw_number)
                res_idx = val[1]
                bw = val[2]
                voxel_res_counts[coord].append((res_idx, bw))
            else:
                # expecting tensor([res_num, res_idx])
                res_idx = int(val[1])
                voxel_res_counts[coord].append(res_idx)

    num_dicts = len(voxel_dicts)
    consensus = {}

    # Compute consensus
    for coord, res_list in voxel_res_counts.items():
        counter = Counter(res_list)
        most_common, count = counter.most_common(1)[0]
        if count / num_dicts >= threshold:
            if use_bw:
                res_idx, bw = most_common
                consensus[coord] = (res_idx, np.round(count / num_dicts, 3), bw)
            else:
                consensus[coord] = (most_common, np.round(count / num_dicts, 3))

    return consensus
