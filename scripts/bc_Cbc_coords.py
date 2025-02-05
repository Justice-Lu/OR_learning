"""
Script for Defining and Filtering Canonical Binding Cavity (Cbc) Coordinates

This script processes structural and functional data to define a **Canonical Binding Cavity (Cbc)**, 
a unified binding cavity derived from multiple olfactory receptors (ORs). The Cbc is defined based on the 
intersection of expanded binding cavity coordinates across a set of ORs. It further filters receptor-specific 
cavity and residue coordinates to retain only those within the defined Cbc.

### Steps:
1. **Input Data Preparation**:
   - Load receptor-specific binding cavity (`bc_cav_coords`), cavity surface (`bc_cavsurf_coords`), and residue coordinates (`bc_res_coords`).
   - Translate receptor names to a standardized DL_OR nomenclature using a mapping file (`Olfr_DL`).

2. **Define ORs for Cbc**:
   - Select predefined ORs (e.g., `Or51e2`, `Or1Ad1`, `Or55B4`) and additional ORs spanning across the odor response metric.
   - Ensure all selected ORs exist in the cavity surface coordinates.

3. **Identify Largest Binding Cavity**:
   - Use DBSCAN clustering to identify the largest binding cavity for each selected OR.

4. **Expand Binding Cavity**:
   - Expand the largest binding cavity for each OR outward by a user-defined distance (`expansion_distance`).

5. **Superimpose and Define Cbc**:
   - Superimpose the expanded cavities for all selected ORs.
   - Define the canonical binding cavity as the intersection of cavity space across at least a threshold percentage of ORs.

6. **Filter Coordinates**:
   - Filter receptor-specific cavity and residue coordinates to retain only those overlapping with the Cbc.

### Key Functions and Parameters:
- **`define_binding_cavity_zone`**:
    - Identifies the largest cavity for each OR and expands it.
    - Inputs:
        - `bc_cavsurf_coords`: Dictionary of cavity surface coordinates for each OR.
        - `expansion_distance`: Distance for cavity expansion.
    - Outputs:
        - Expanded coordinates for each OR.
        - Largest cavity coordinates for each OR.
  
- **`filter_coordinates_within_cavity`**:
    - Filters 3D coordinates (or residue-based coordinates) to include only those within a specified cavity zone.
    - Inputs:
        - `cavity_zone`: Canonical binding cavity coordinates.
        - `coordinates`: Receptor-specific coordinates to filter.
        - `is_residue`: Boolean flag to handle residue-based filtering.
    - Outputs:
        - Filtered coordinates.

### Input Files:
1. **`dict_bc_pyKVcav_tmaligned.pkl`**:
    - Pickle file containing cavity coordinates for each receptor.
2. **`dict_bc_pyKVcavsurf_tmaligned.pkl`**:
    - Pickle file containing cavity surface coordinates for each receptor.
3. **`dict_bc_pyKVres_tmaligned.pkl`**:
    - Pickle file containing residue coordinates for each receptor.
4. **`Olfr_DL.csv`**:
    - CSV file mapping olfactory receptor names to DL_OR nomenclature.
5. **`compiled_odor_sigResp_wide.csv`**:
    - CSV file containing odor response metrics for each receptor.

### Output Files:
1. **`dict_Cbc_cav_coords.pkl`**:
    - Pickle file containing filtered cavity surface coordinates within the Cbc for each receptor.
2. **`dict_Cbc_res_coords.pkl`**:
    - Pickle file containing filtered residue coordinates within the Cbc for each receptor.

### Parameters:
- **`expansion_distance`**: 
    - Distance by which to expand the binding cavity. Default is 3.0.
- **`threshold`**: 
    - Minimum fraction of ORs' expanded cavity overlap to define the Cbc. Default is 10% (0.1).

### Example Use Case:
1. Preprocess and cluster cavity surface data using DBSCAN.
2. Expand binding cavities to ensure coverage.
3. Define the canonical binding cavity by intersection thresholds.
4. Filter receptor-specific cavity and residue coordinates for downstream analysis.

### Notes:
- **Canonical Binding Cavity (Cbc)**:
    - Defined as the intersection of binding cavities across multiple ORs.
    - Captures binding cavity regions intersecting at least a threshold percentage of cavities.

- **Clustering**:
    - DBSCAN parameters (`eps`, `min_samples`) should be tuned based on the scale of input data.

- **Performance**:
    - Coordinate trimming is performed using integer rounding (`// 1`) to reduce computation time when processing large datasets.
"""

import numpy as np 
import pandas as pd 
import sys 
from tqdm import tqdm 
import pickle 


sys.path.append('/data/jlu/OR_learning/utils/')

import BindingCavity_functions as bc 
import voxel_functions as vf
import color_function as cf 
import plot_functions as pf 


# Read pickle file for data 
bc_cav_coords = pd.read_pickle('/data/jlu/OR_learning/files/dict_bc_pyKVcav_tmaligned.pkl')
bc_cavsurf_coords = pd.read_pickle('/data/jlu/OR_learning/files/dict_bc_pyKVcavsurf_tmaligned.pkl')
bc_res_coords = pd.read_pickle('/data/jlu/OR_learning/files/dict_bc_pyKVres_tmaligned.pkl')

# Create cavity voxels from coordinates 
Olfr_DL = pd.read_csv('/data/jlu/OR_learning/files/Olfr_DL.csv', index_col=0)
Olfr_DL = dict(zip(Olfr_DL.Olfr, Olfr_DL.DL_OR))

# Translate to DL_OR
bc_cav_coords = {Olfr_DL.get(_olfr, _olfr): bc_cav_coords[_olfr] for _olfr in bc_cav_coords.keys()}
bc_res_coords = {Olfr_DL.get(_olfr, _olfr): bc_res_coords[_olfr] for _olfr in bc_res_coords.keys()}
bc_cavsurf_coords = {Olfr_DL.get(_olfr, _olfr): bc_cavsurf_coords[_olfr] for _olfr in bc_cavsurf_coords.keys()}


odorResp = pd.read_csv('../../files/Deorphanization/compiled_odor_sigResp_wide.csv', index_col=0)
odorResp.columns = [Olfr_DL.get(_col, _col) for _col in odorResp] # Change to DL_OR


from sklearn.cluster import DBSCAN


DEFINE_CAVZONE_OR = ['Or51e2', 'Or1Ad1', 'Or55B4'] + list(odorResp.sum().sort_values().index[::10])
DEFINE_CAVZONE_OR = [_Or for _Or in DEFINE_CAVZONE_OR if _Or in list(bc_cavsurf_coords.keys())]

# Getting expanded coords for defined ORs 
expanded_coords, largest_cavity_coords = bc.define_binding_cavity_zone({_Or : bc_cavsurf_coords[_Or] for _Or in DEFINE_CAVZONE_OR}, 
                                                                       expansion_distance=3)

# Trim coordinates to speed up procees in making voxel 
expanded_coords_trimmerd = {}
for _Or in expanded_coords: 
    expanded_coords_trimmerd[_Or] = np.unique(expanded_coords[_Or] // 1, axis=0)
    
# Identify Canonical Binding Cavity by threshold intersection of expanded cavity coordinates 
threshold = 0.1
min_count = int(threshold * len(expanded_coords_trimmerd))
all_coordinates = np.vstack(list(expanded_coords_trimmerd.values()))
unique_coordinates, counts = np.unique(all_coordinates, axis=0, return_counts=True)

canonical_bc_coords = unique_coordinates[counts >= min_count]

# Filtering ORs cavity and residue coordinates if they overlap with the defined canonical binding cavity
Cbc_cav_coords_ = { _Or: bc.filter_coordinates_within_cavity(canonical_bc_coords, 
                                                             np.array(bc_cavsurf_coords[_Or])) for _Or in tqdm(bc_cavsurf_coords)}
with open('/data/jlu/OR_learning/files/dict_Cbc_cav_coords.pkl', 'wb') as f:
    pickle.dump(Cbc_cav_coords_, f)

Cbc_res_coords = { _Or: bc.filter_coordinates_within_cavity(canonical_bc_coords, 
                                                             np.array(bc_res_coords[_Or]), 
                                                             is_residue=True) for _Or in tqdm(bc_res_coords)}
with open('/data/jlu/OR_learning/files/dict_Cbc_res_coords.pkl', 'wb') as f:
    pickle.dump(Cbc_res_coords, f)
