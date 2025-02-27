"""
Script for Running PyKVFinder on AlphaFold PDB Files to Identify Binding Cavities

This script automates the process of identifying and extracting binding cavity 
coordinates from AlphaFold-predicted protein structures. The script uses 
PyKVFinder to analyze binding cavities in each PDB file and transforms the 
voxel-based results into 3D coordinates aligned with the original PDB space. 

The final output is a dictionary where each key corresponds to a protein's 
Doron Lancet Olfactory Receptor (DL_OR) nomenclature, and each value contains 
information about its binding cavities.

Workflow:
1. Define paths to AlphaFold PDB files and supporting metadata.
2. Run PyKVFinder on each relevant PDB file to identify binding cavities.
3. Transform the PyKVFinder voxel-based results into 3D coordinates.
4. Store the results in a dictionary using DL_OR nomenclature.
5. Save the dictionary as a pickle file for later use.

Modules:
- numpy: Used for numerical operations.
- pandas: For handling metadata about proteins (e.g., Olfr_DL).
- pickle: To save and load the dictionary.
- os: For file path operations.
- sys: To append custom module paths.
- tqdm: For progress tracking.
- pyKVFinder: To identify and analyze binding cavities.
- BindingCavity_functions: Custom utility functions for processing cavity data.

Outputs:
- A dictionary (saved as a pickle file) with keys corresponding to protein DL_OR 
  names and values containing binding cavity coordinates, centers, and min/max bounds.
"""

import numpy as np 
import pandas as pd 
import pickle 
import os 
import sys 
from tqdm import tqdm 

sys.path.append('/data/jlu/OR_learning/utils/')
import BindingCavity_functions as bc 

import pyKVFinder
from sklearn.cluster import DBSCAN


# Define paths of AF files 
AF2_PATH = '/data/jlu/AF_files/AF_tmaligned_pdb'
pdb_files = os.listdir(AF2_PATH)
Olfr_DL = pd.read_csv('/data/jlu/OR_learning/files/Olfr_DL.csv', index_col=0)
Olfr_DL = dict(zip(Olfr_DL.Olfr, Olfr_DL.DL_OR))


from tqdm import tqdm

# Running pyKVFinder standard workflow for cavity grid
bc_cav_coords = {}
bc_cavsurf_coords = {}
bc_res_coords = {}

for _pdb in tqdm(pdb_files): 
    _olfr = _pdb.split('_')[0]
    bc_results = pyKVFinder.run_workflow(os.path.join(AF2_PATH, _pdb))
    bc_atomic = pyKVFinder.read_pdb(os.path.join(AF2_PATH, _pdb))
    
    if bc_results == None: 
        print(f"NOT GENERATED . . . {_olfr}..........................................")
        continue 
    
    # Store cavity coordinates in dict with DL_OR name
            # Store cavity coordinates in dict with DL_OR name
    combined_coords = []
    bc_results_coord = bc.grid2coords(bc_results)
    # Append cavity coordinates into dict 
    for _cav_coords in bc_results_coord[0].values(): 
        combined_coords.extend(_cav_coords)
    bc_cav_coords[Olfr_DL.get(_olfr, _olfr)] = combined_coords
    
    combined_coords = []
    # Append cavity surface coordinates into dict 
    for _cavsurf_coords in bc_results_coord[1].values(): 
        combined_coords.extend(_cavsurf_coords)
    bc_cavsurf_coords[Olfr_DL.get(_olfr, _olfr)] = combined_coords
    
    
    # Get cavity interacting residue coords 
    res_coords = []
    res_coords_dict = bc.res2atomic(bc_results, bc_atomic)
    for _res in res_coords_dict: 
        res_coords.extend(res_coords_dict[_res][:, [0,2,3,4,5,6]].tolist())  # Extract [ResNum, AA, Atom, x, y, z]
    # remove duplicated residues 
    bc_res_coords[Olfr_DL.get(_olfr, _olfr)] = [list(x) for x in set(tuple(entry) for entry in res_coords)]
           

# Save dict as pkl 
with open('/data/jlu/OR_learning/files/dict_bc_pyKVcav_tmaligned.pkl', 'wb') as f:
    pickle.dump(bc_cav_coords, f)
with open('/data/jlu/OR_learning/files/dict_bc_pyKVcavsurf_tmaligned.pkl', 'wb') as f:
    pickle.dump(bc_cavsurf_coords, f)
with open('/data/jlu/OR_learning/files/dict_bc_pyKVres_tmaligned.pkl', 'wb') as f:
    pickle.dump(bc_res_coords, f)


