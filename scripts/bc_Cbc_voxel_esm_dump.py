
"""
Generate voxel representations of binding cavities and surrounding residues for ORs using ESM-encoded residue features.

This script performs the following steps:
1. Loads residue-level ESM embeddings and corresponding OR labels.
2. Loads pre-processed cavity and residue 3D coordinates for a set of ORs.
3. Matches ESM embeddings to each OR.
4. Voxelizes both cavity and residue regions into a 4D tensor (X, Y, Z, D), where D is the feature dimension.
5. Saves the resulting list of voxel grids (`Cbc_voxels`) to disk using pickle.

The resulting voxelized data can be used as input for 3D deep learning models for ligand-binding prediction or OR classification.

Dependencies:
- `voxel_functions.voxelize_cavity` must support ESM encoding mode and tensor outputs.

Output:
- list_Cbc_voxels_esm.pkl: List of voxelized tensors with ESM features for each OR in `Cbc_cav_coords`.
"""

import numpy as np 
import pandas as pd 
import os 
import sys 

OR_LEARNING_PATH = os.path.join(os.getcwd().split('OR_learning')[0], 'OR_learning/')
sys.path.insert(0, os.path.join(OR_LEARNING_PATH, 'utils/'))

import voxel_functions as vf

res_esm = np.load('/mnt/data2/Justice/OR_learning/files/ESM/residue_embeddings.npy', 
                  allow_pickle=True)
OR_label = np.load('/mnt/data2/Justice/OR_learning/files/ESM/esm_OR_order.npy', 
                 allow_pickle=True)

# Join OR label and ESM 
OR_esm = {str(_OR): res_esm[i] for i, _OR in enumerate(OR_label)}

Cbc_cav_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/dict_Cbc_cav_coords.pkl')
Cbc_res_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/dict_Cbc_res_coords.pkl')


from tqdm import tqdm 

sparse_voxels, grid_shape = vf.voxelize_cavity(cavity_coords = Cbc_cav_coords,
                                      residue_coords = Cbc_res_coords,
                                      resolution = 1, 
                                      encode_method='esm', 
                                      OR_esm = OR_esm, 
                                      esm_order = list(Cbc_cav_coords), 
                                      vdw_radius = True, 
                                      sparse_mode = True)

import torch 
all_pooled = []
for sparse_vox in tqdm(sparse_voxels):  # assume you saved grid_shapes
    pooled = vf.sparse_avg_pool3d(sparse_vox, grid_shape, kernel_size=4, stride=4)
    all_pooled.append(pooled.flatten())  # flatten 3D → 1D vector per structure


from sklearn.decomposition import PCA

# Stack pooled features into a matrix of shape (n_structures, pooled_dim)
X = torch.stack(all_pooled)  # Shape: (n_structures, pooled_dim)

# Identify columns that are constant across all proteins (e.g., always zero)
mask = X.std(dim=0) > 0  # Exclude dimensions with no atoms across structures

# Apply the mask to reduce dimensionality
X_filtered = X[:, mask]  # shape: (N, D_reduced)

# Run PCA
pca = PCA(n_components=2)  # Or 3 for 3D plotting
X_pca = pca.fit_transform(X_filtered)
variance_ratio = pca.explained_variance_ratio_

pd.DataFrame(X_pca, index=list(Cbc_cav_coords)).to_csv('/mnt/data2/Justice/OR_learning/output/ESM/Cbc_esm_sparse_voxels/pca_df.csv')

