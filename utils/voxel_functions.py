import pandas as pd 
import numpy as np 
import os 
import copy
from tqdm import tqdm


"""
Takes in dict generated from reading PDB to generate information neede for making voxels. 

dict = {'Olfr0' : {'atom': np.array([],
                     'coord': np.array([],
                     'resid': np.array(), 
                     'amino_acid': np.array()
                     },
        'Olfr1' : ...
        }
"""
def PDB_voxel_info(PDB_dict, resolution = 1):
    # Find the maximum extent of all the proteins
    max_extent = np.max([np.max(PDB_dict[Olfr]['coord'], axis=0) - \
                         np.min(PDB_dict[Olfr]['coord'], axis=0) \
                         for Olfr in PDB_dict], axis=0)
    min_spacer = np.min([np.min(PDB_dict[Olfr]['coord'], axis=0) \
                         for Olfr in PDB_dict], axis=0)
    voxel_shape = np.ceil((max_extent - min_spacer )/ resolution).astype(int)
    
    return max_extent, min_spacer, voxel_shape
    


"""
Create a voxel from pdb coordinates. 
coords - PDB coordinate
resolution - the resolution of the voxel in Å. (Scales up exponentially) 
"""
def create_voxel(PDB_dict,  
                 resolution, 
                 fill_radius=False):
    
    max_extend, min_spacer, voxel_shape = PDB_voxel_info(PDB_dict, resolution = resolution)
    # Calculate the voxel shape based on the maximum and minimum coordinates
    voxel_shape = voxel_shape + np.ceil((max_extend - min_spacer) / resolution).astype(int)
    
    voxel_list = []    
    Olfr_order = sorted(PDB_dict.keys())
    for Olfr in tqdm(Olfr_order): 
        # Initialize the voxel
        voxel = np.zeros(np.array(list(voxel_shape) + [len(ATOM_ENCODING)]), dtype=int)
        
        # Compute the indices of the coordinates in the voxel
        # indices = np.floor((PDB_dict[Olfr]['coord'] - min_spacer) / resolution).astype(int)
        

        # Set the values of the voxel based on radii
        if fill_radius: 
            for idx, index in enumerate(indices):
                voxel_index = tuple(index)
                radius = ATOM_RADIUS_DICT[PDB_dict[Olfr]['atom'][idx]]
                num_points = int(np.ceil(radius / resolution))
                assign_voxel(voxel, voxel_index, num_points)
        else: 
            # Previously does not discriminate between atoms
            # voxel[indices[:,0], indices[:,1], indices[:,2]] = 1
            
            # Get indice for each atom and assign OHE vector
            for atom in ATOM_ENCODING.keys():
                atom_indice = PDB_dict[Olfr]['coord'][np.where(PDB_dict[Olfr]['atom'] == atom)]
                indices = np.floor((atom_indice - min_spacer) / resolution).astype(int)
                voxel[indices[:,0], indices[:,1], indices[:,2]] = ATOM_ENCODING[atom]
                # Line to check to prevent loss of resolution when scaled 
                # print(f'indices {len(indices)}... num_pos {len(np.argwhere(np.any(voxel, axis=3)))}')
            # When using res = 1, there seems to be cost of accuracy, as there will be close points that become the same position
            # print(f"{len(PDB_dict[Olfr]['coord'])}...{len(np.argwhere(np.any(voxel, axis=3)))}")

         # Save the voxel to a file or do other processing as needed
        voxel_list.append(voxel)
    return voxel_list, voxel_shape, Olfr_order

"""
Called by create_voxel. 
When fill_radius=True. 
Calls for assign_voxel, to fill in coordinates in voxels that are within the radius of the coordinate
"""
def assign_voxel(voxel, voxel_index, num_points):
    for i in range(-num_points, num_points + 1):
        for j in range(-num_points, num_points + 1):
            for k in range(-num_points, num_points + 1):
                distance = np.sqrt(i**2 + j**2 + k**2)
                if distance <= num_points:
                    try:
                        voxel[voxel_index[0] + i, voxel_index[1] + j, voxel_index[2] + k] = 1
                    except IndexError:
                        continue
# OLD FUNCTIONING create_voxel ABOVE MODIFICATION ADDS RADIUS 
# def create_voxel(coords, voxel_shape, resolution, spacer = [0,0,0]):
#     # Initialize the voxel
#     voxel = np.zeros(voxel_shape, dtype=int)
#     # Compute the indices of the coordinates in the voxel
#     indices = np.floor((coords - spacer) / resolution).astype(int)
#     # Set the values of the voxel
#     voxel[indices[:,0], indices[:,1], indices[:,2]] = 1
#     return voxel


"""
Get top features from a list of flattened_voxel_values 
"""
def get_top_features(features, num_features = 10):

    FEATURE_LIST = []
    for i in features: 
        indexed_feature = list(enumerate(i))
        sorted_feature = sorted(indexed_feature, key=lambda x: x[1], reverse=True)
        
        features = []
        max_y = sorted_feature[0][1]
        count = 0

        for x, y in sorted_feature:
            if y == max_y or count < num_features:
                features.append((x, y))
                count += 1
            elif y < max_y:
                break
                
        FEATURE_LIST.append(features)
    
    return FEATURE_LIST

"""
get_top_differnce function specifically takes in 2 lists of flatten voxels to conduct pairewise difference. 
It simply finds the index with the maximum difference by substracting indices 
"""
def get_top_difference(list_a, list_b, num_features=10):
    
    if len(list_a) != len(list_b):
        raise ValueError("Both lists must have the same length.")
        
    # Calculate the absolute differences between corresponding elements
    differences = [abs(a - b) for a, b in zip(list_a, list_b)]

    # Create a list of tuples containing the differences and corresponding indices
    indexed_differences = list(enumerate(differences))
    # Sort the indexed differences based on the differences in descending order
    sorted_differences = sorted(indexed_differences, key=lambda x: x[1], reverse=True)

    # Get the top x differences and their indices
    features = []
    max_y = sorted_differences[0][1]
    count = 0

    for x, y in sorted_differences:
        if y == max_y or count < num_features:
            features.append((x, y))
            count += 1
        elif y < max_y:
            break

    return features

"""
Get 3d voxel coordinate from flattened voxel indices. 

cluster_voxel_data contains list of kClusters
Within that list contains 4 np.arrays()
[0,1,2] - voxel indice location 
[4] - the percentage shared by Olfrs within the cluster 
"""
def get_3Dcoord(features, pos_space, voxel_shape, max_scale=20, min_scale=10):

# For testing
# random_indice = np.array([random.randint(0, len(flat_voxel_data[0])) for _ in range(10000)])

    FEATURE_3Dcoord = []
    for feature in features:
        indices, percent_shared = zip(*feature)

        indices_3d = np.unravel_index(pos_space[list(indices)], voxel_shape) 
        indices_3d = list(indices_3d)
    #     Add percent_shared information. Scale between 10 and 5 for size plotting
        indices_3d.append(np.array(scale(percent_shared, scale_between=[max_scale,
                                                                        min_scale])))
        FEATURE_3Dcoord.append(indices_3d)
    return FEATURE_3Dcoord
        
"""
Reverses the scale of a list of values such that the smallest value becomes 1 and the largest value becomes 0.
"""        
def scale(values, reverse=False, factor = 1, scale_between = [1,0]):
    min_val = min(values)
    max_val = max(values)
    new_max = scale_between[0]
    new_min = scale_between[1]

    if min_val != max_val: #if all the min and max is the same value. assign  max size 
        scaled_values = [(value - min_val) * (new_max - new_min) / (max_val - min_val) + new_min for value in values]
    else: 
        scaled_values = [value*new_max for value in values]
#     scaled_values = [(val - min_val) / (max_val - min_val)*factor for val in values]
    if reverse:
        scaled_values = [1 - val for val in scaled_values]
    return scaled_values


ATOM_RADIUS_DICT = {
    'C': 1.70, 'CA': 1.80, 'CB': 1.90, 'CD': 1.88, 'CD1': 1.88, 'CD2': 1.88, 
    'CE': 1.88, 'CE1': 1.88, 'CE2': 1.88, 'CE3': 1.88, 'CG': 1.88, 'CG1': 1.88 ,
    'CG2': 1.88, 'CH2': 1.88, 'CZ': 1.88, 'CZ2': 1.88, 'CZ3': 1.88,
    'N': 1.55, 'ND1': 1.55, 'ND2': 1.55, 'NE': 1.55, 'NE1': 1.55, 'NE2': 1.55,
    'NH1': 1.55, 'NH2': 1.55, 'NZ': 1.55,
    'O': 1.40, 'OD1': 1.40, 'OD2': 1.40, 'OE1': 1.40, 'OE2': 1.40, 'OG': 1.40 ,
    'OG1': 1.40, 'OH': 1.40, 'OXT': 1.40, 'SD': 2.00, 'SG': 1.80}

ATOM_ENCODING = {'C': [1, 0, 0, 0], 
                 'N': [0, 1, 0, 0], 
                 'O': [0, 0, 1, 0], 
                 'S': [0, 0, 0, 1]}



def _res_coord_voxel_prep(res_data): 
    """
    Translates residue coordinates into a list of coordinates and their corresponding residue classes.
    
    Each residue is assigned to one of the following classes:
    - Alipathic apolar (class 2): Alanine, Glycine, Isoleucine, Leucine, Methionine, Valine
    - Aromatic (class 3): Phenylalanine, Tryptophan, Tyrosine
    - Polar uncharged (class 4): Asparagine, Cysteine, Glutamine, Proline, Serine, Threonine
    - Negatively charged (class 5): Aspartate, Glutamate
    - Positively charged (class 6): Arginine, Histidine, Lysine
    - Non-standard (class 7): Non-standard residues
    
    Parameters:
    -----------
    data : list
        A list of lists, where each sublist contains residue information in the format:
        [residue_number, residue_name, atom_name, x, y, z]
    
    Returns:
    --------
    list
        A list of lists, where each sublist contains:
        [x, y, z, [0, 0, 0, 0, 0, 0, 0, 0]] where the 1-based index of the residue class is set to 1.
    
    Example:
    --------
    Input:
    [['24', 'GLU', 'OE2', '27.382', '-3.966', '-2.635']]
    
    Output:
    [[27.382, -3.966, -2.635, [0, 0, 0, 0, 1, 0, 0, 0]]]
    """
    
    # Define residue classes
    residue_classes = {'ALA': 2,'GLY': 2,'ILE': 2,'LEU': 2,'MET': 2,'VAL': 2,
                       'PHE': 3,'TRP': 3,'TYR': 3,
                       'ASN': 4,'CYS': 4,'GLN': 4,'PRO': 4,'SER': 4,'THR': 4,
                       'ASP': 5,'GLU': 5,
                       'ARG': 6,'HIS': 6,'LYS': 6,
                       'UNK': 7,'NON': 7}

    translated_data = []
    for _olfr in res_data:
        _olfr_res = []
        for _res in _olfr: 
            x, y, z = map(float, _res[3:6])  # Extract coordinates and convert to float
            residue = _res[1]  # Extract residue name
            cls = residue_classes.get(residue, 7)  # Default to class 7 if residue not found
            class_vector = [0] * 8  # Initialize all-zero vector
            class_vector[cls] = 1  # Set appropriate class index to 1
            _olfr_res.append([x, y, z, class_vector])  # Append the result
        translated_data.append(_olfr_res)
    return translated_data

def voxelize_cavity(cavity_coords, 
                    residue_coords=None, 
                    resolution=1):
    """
    Voxelizes 3D coordinates by creating a voxel grid representation.
    If residue_coords are provided, each voxel will store an 8-element one-hot encoded array.
    If residue_coords are not provided, each voxel will store a single value: 0 (empty) or 1 (occupied).
    
    :param cavity_coords: 
        List of lists containing 3D coordinates for cavities.
        Example: 
        [
            [array([x1, y1, z1]), array([x2, y2, z2])],
            [array([x3, y3, z3]), array([x4, y4, z4])]
        ]
    
    :param residue_coords: 
        List of lists containing residue interaction data with one-hot encoded classes.
        Each residue coordinate must be in the form:
        [residue_number, residue_name, atom_name, x, y, z].
        Example:
        [['24', 'GLU', 'OE2', '27.382', '-3.966', '-2.635']]

        Default is None. If None, voxel grid will store a single value at each position.

    :param resolution: 
        Size of each voxel (default is 1).
        Determines the granularity of the voxel grid.
    
    :return: 
        - List of voxel grids for each cavity. 
          Each grid is a numpy array of shape X x Y x Z x N, where N = 8 (if residue_coords is supplied)
          or N = 1 (if residue_coords is None).
        - Tuple of the grid shape (X, Y, Z).
    """
    # Handle residue coordinate preparation if supplied
    residue_coords_class = _res_coord_voxel_prep(residue_coords) if residue_coords else None

    # Gather all coordinates (cavity and residue) for defining grid boundaries
    all_coords = list(np.concatenate(cavity_coords, axis=0))
    if residue_coords_class:
        all_coords.extend([_coord[:3] for _all_coord in residue_coords_class for _coord in _all_coord])
    
    # Compute grid boundaries
    min_coords = np.min(all_coords, axis=0)
    max_coords = np.max(all_coords, axis=0)
    
    # Define voxel grid shape
    grid_shape = np.ceil((max_coords - min_coords) / resolution).astype(int) + 1
    
    # Initialize voxelized data
    voxelized_data = []
    
    for i, cavity in enumerate(cavity_coords):
        if residue_coords_class and i < len(residue_coords):
            # 8-dimensional grid for one-hot encoded data
            voxel_grid = np.zeros((*grid_shape, 8), dtype=int)
            coords_data = [[_coord[0], _coord[1], _coord[2], [1, 0, 0, 0, 0, 0, 0, 0]] for _coord in cavity]
            coords_data.extend(residue_coords_class[i])
        else:
            # 1-dimensional grid for binary data
            voxel_grid = np.zeros((*grid_shape, 1), dtype=int)
            coords_data = [[_coord[0], _coord[1], _coord[2], [1]] for _coord in cavity]
        
        # Map coordinates to voxel grid
        for point in coords_data:
            grid_x = int((point[0] - min_coords[0]) // resolution)
            grid_y = int((point[1] - min_coords[1]) // resolution)
            grid_z = int((point[2] - min_coords[2]) // resolution)
            
            # Update voxel grid with the corresponding value (one-hot or binary)
            voxel_grid[grid_x, grid_y, grid_z] = point[3]
        
        # Append the voxel grid for the current cavity
        voxelized_data.append(voxel_grid)
    
    return voxelized_data, grid_shape


def voxelize_coordinates(cavity_coords, resolution=1):
    """
    ********** DEPRECATED ********** 
    USE voxelize_cavity INSTEAD. 
    
    Voxelizes the 3D coordinates by placing 1s in voxels that are occupied by coordinates.
    
    :param cavity_coords: List of lists containing 3D coordinates for each Olfr
    :param resolution: Size of each voxel (default is 1)
    :return: List of 1D arrays representing the voxelized space for each Olfr
    """
    # Step 1: Find the global min and max coordinates across all cavities
    all_coords = np.concatenate(cavity_coords, axis=0)
    min_coords = np.min(all_coords, axis=0)
    max_coords = np.max(all_coords, axis=0)
    
    # Step 2: Define voxel grid shape
    grid_shape = np.ceil((max_coords - min_coords) / resolution).astype(int)
    
    # Step 3: Create a 3D grid for each cavity
    voxelized_data = []
    
    for cavity in cavity_coords:
        # Step 4: Create an empty voxel grid
        voxel_grid = np.zeros(grid_shape, dtype=int)
        
        # Step 5: Convert each cavity point to voxel grid indices
        for point in cavity:
            # Translate point into grid coordinates
            grid_x = int((point[0] - min_coords[0]) // resolution)
            grid_y = int((point[1] - min_coords[1]) // resolution)
            grid_z = int((point[2] - min_coords[2]) // resolution)
            
            # Mark the voxel as occupied
            voxel_grid[grid_x, grid_y, grid_z] = 1
        
        voxelized_data.append(voxel_grid)
    
    return voxelized_data, grid_shape

