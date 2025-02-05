
import numpy as np 
import pandas as pd 
import sys 

sys.path.append('/data/jlu/OR_learning/utils/')

import voxel_functions as vf


# Read pickle file for data 
bc_cav_coords = pd.read_pickle('/data/jlu/OR_learning/files/dict_bc_pyKVcav_tmaligned.pkl')
bc_res_coords = pd.read_pickle('/data/jlu/OR_learning/files/dict_bc_pyKVres_tmaligned.pkl')

# Create cavity voxels from coordinates 
# DROP Or defined in the exclusion list below
EXCLUDE_OR_LIST = ['Or4Q3', 'Or2W25', 'Or2l1', 'Or4A67', 'Or2I1']
bc_cav_coords = {key: value for key, value in bc_cav_coords.items() if key not in EXCLUDE_OR_LIST}
bc_res_coords = {key: value for key, value in bc_res_coords.items() if key not in EXCLUDE_OR_LIST}

# DROP non DL_OR names
bc_cav_coords = {key: value for key, value in bc_cav_coords.items() if key.startswith('Or')}
bc_res_coords = {key: value for key, value in bc_res_coords.items() if key.startswith('Or')}

# Voxelize binding cavity coordinates 
# voxelized_cavities, voxel_shape = vf.voxelize_cavity(list(bc_cav_coords.values()), resolution=1)

voxelized_cav_res, voxel_shape = vf.voxelize_cavity(list(bc_cav_coords.values()), 
                                                     list(bc_res_coords.values()), resolution=1)
print('Voxel Generated . . .')

# Output: List of 1D arrays representing voxelized space
print(np.array(voxelized_cav_res).shape)
# pd.to_pickle(voxelized_cav_res, '../../output/bc_voxel/voxelized_cav_res.pkl')


# Save corr_matrix for downstream comparison 
flattened_voxels = [voxel.flatten() for voxel in voxelized_cav_res]
print('Voxel Flattened . . .')

# Convert to a NumPy array for efficient computation
flattened_voxels_array = np.array(flattened_voxels)
# Identify attributes (columns) with zero variance
non_zero_variance_mask = np.var(flattened_voxels_array, axis=0) > 0
# Filter out zero-variance attributes
filtered_voxels = flattened_voxels_array[:, non_zero_variance_mask]
print(f'Voxel Filterred from {len(flattened_voxels_array[0])} to {len(filtered_voxels[0])}')

corr_matrix = np.corrcoef(filtered_voxels)
corr_df = pd.DataFrame(corr_matrix, 
             columns=bc_cav_coords.keys(), 
             index=bc_cav_coords.keys())

print('corr_df Generated . . .')
corr_df.to_csv('/data/jlu/OR_learning/output/binding_cavity/Correlation/cav_res_corr_df.csv', index=0)

print('csv SAVED . . .')