import pandas as pd 
import numpy as np 
import os 
import sys 

sys.path.append('/data/jlu/OR_learning/utils/')
import BindingCavity_functions as bc 
import voxel_functions as vf
import color_function as cf 

"""
Script for Generating a Jaccard Similarity Matrix for Binding Cavity Voxelized Representations

This script processes a dictionary of binding cavity coordinates, voxelizes them into a binary representation, 
and computes a pairwise Jaccard similarity matrix between the voxelized cavities. The results are saved as 
NumPy arrays for further analysis.

Modules:
- `pickle`: Used for loading the input dictionary from a pickle file.
- `numpy`: For numerical operations and array manipulations.
- `pandas`, `os`, `sys`: Utility modules for file operations and system path management.
- `tqdm`: Displays progress bars during computations.
- `sklearn.metrics.jaccard_score`: Computes the Jaccard similarity between binary arrays.

External Imports:
- `BindingCavity_functions`, `voxel_functions`, `color_function`: Custom modules for handling binding cavity 
  data, voxelization, and color mapping.

Steps:
1. **Load Binding Cavity Dictionary**:
   - Loads a pickle file containing binding cavity data with coordinates, centers, and min-max bounds.

2. **Process Coordinates**:
   - Converts the dictionary format from nested entries into a single list of coordinates for each key.
   - Re-orders the dictionary based on OR names in ascending order.

3. **Exclude Unwanted ORs**:
   - Removes entries specified in `EXCLUDE_OR_LIST`.
   - Keeps only entries whose keys start with "Or".

4. **Voxelize Binding Cavity Coordinates**:
   - Converts 3D coordinates into binary voxelized representations using the `voxelize_coordinates` function 
     with a resolution of 1 unit.

5. **Compute Jaccard Similarity**:
   - Flattens each voxelized 3D array into a 1D array for pairwise comparison.
   - Initializes a symmetric Jaccard similarity matrix.
   - Computes the Jaccard index for all pairwise combinations of voxels and populates the matrix.

Outputs:
- `jaccard_matrix.npy`: The computed Jaccard similarity matrix.
- `jaccard_ORkeys.npy`: Array of OR keys corresponding to the rows and columns of the similarity matrix.

Usage:
Run the script in an environment where the required modules and data files are available. 
Ensure the paths for input pickle files and output directories are correctly specified.

"""
# Open pickle file of binding cavity dictionary
import pickle 

file = open('/data/jlu/OR_learning/files/dict_bc_cav_tmaligned.pkl', 'rb')
# dump information to that file
bc_cav_coords = pickle.load(file)
file.close()


# Convert from bc.grid2coords output format for each entry of coords, center, minmax to only coords
# Additionally combines all cavity coords into a single array instead of separate key for individual cavity 
for key in bc_cav_coords:
    combined_coords = []
    for subkey in bc_cav_coords[key][0]:
        combined_coords.extend(bc_cav_coords[key][0][subkey])
    bc_cav_coords[key] = combined_coords
    
# Re-order based on OR names
bc_cav_coords = {i: bc_cav_coords[i] for i in np.sort(list(bc_cav_coords.keys()))}


# DROP Or defined in the exclusion list below
EXCLUDE_OR_LIST = ['Or4Q3', 'Or2W25', 'Or2l1', 'Or4A67', 'Or2I1']
bc_cav_coords = {key: value for key, value in bc_cav_coords.items() if key not in EXCLUDE_OR_LIST}
# DROP non DL_OR names
bc_cav_coords = {key: value for key, value in bc_cav_coords.items() if key.startswith('Or')}


# Voxelize binding cavity coordinates 
voxelized_cavities, voxel_shape = vf.voxelize_coordinates(list(bc_cav_coords.values()), resolution=1)

# Output: List of 1D arrays representing voxelized space
print(np.array(voxelized_cavities).shape)



from tqdm import tqdm
from sklearn.metrics import jaccard_score


# Example: voxelized_cavities is a list of 3D binary arrays (0s and 1s)
# Flatten each voxel array into a 1D array for pairwise comparison
flattened_voxels = [voxel.flatten() for voxel in voxelized_cavities]

# Initialize an empty matrix to store Jaccard similarity scores
num_voxels = len(flattened_voxels)
jaccard_matrix = np.zeros((num_voxels, num_voxels))

# Compute pairwise Jaccard index
for i in tqdm(range(num_voxels)):
   for j in range(i, num_voxels):  # Only compute upper triangle (matrix is symmetric)
        jaccard_index = jaccard_score(flattened_voxels[i], flattened_voxels[j])
        jaccard_matrix[i, j] = jaccard_index
        jaccard_matrix[j, i] = jaccard_index  # Symmetric entry
        
   if i % 50 == 0: # Save every 50 increments 
      # Save jaccard_comparison index 
      np.save('../output/binding_cavity/Jaccard/jaccard_matrix.npy', jaccard_matrix)
      np.save('../output/binding_cavity/Jaccard/jaccard_ORkeys.npy', np.array(list(bc_cav_coords.keys())))

# Save jaccard_comparison index 
np.save('../output/binding_cavity/Jaccard/jaccard_matrix.npy', jaccard_matrix)
np.save('../output/binding_cavity/Jaccard/jaccard_ORkeys.npy', np.array(list(bc_cav_coords.keys())))