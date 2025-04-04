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

    This function is designed for classifying residue properties based on their chemical characteristics.
    Amino acid classification starts at index **1** because index 0 is reserved to indicate whether a position 
    represents a cavity space (1) or not (0). If the value is 0, the position may correspond to an amino acid 
    or an unoccupied space.

    Each residue is assigned to one of the following classes:
    - Cavity space (index 0): Indicates the presence of a cavity (binary flag at index 0)
    - Aliphatic apolar (index 1): Alanine, Glycine, Isoleucine, Leucine, Methionine, Valine
    - Aromatic (index 2): Phenylalanine, Tryptophan, Tyrosine
    - Polar uncharged (index 3): Asparagine, Cysteine, Glutamine, Proline, Serine, Threonine
    - Negatively charged (index 4): Aspartate, Glutamate
    - Positively charged (index 5): Arginine, Histidine, Lysine
    - Non-standard (index 6): Non-standard residues
    
    Parameters:
    -----------
    res_data : list
        A list of lists, where each sublist contains residue information in the format:
        [residue_number, residue_name, atom_name, x, y, z]
    
    Returns:
    --------
    list
        A list of lists, where each sublist contains:
        [x, y, z, [0, 0, 0, 0, 0, 0, 0]] where the corresponding index of the residue class is set to 1.
        The first index (0) indicates cavity presence (1 for cavity, 0 otherwise), while residue classifications
        start at index 1.

    Example:
    --------
    Input:
    [['24', 'GLU', 'OE2', '27.382', '-3.966', '-2.635']]
    
    Output:
    [[27.382, -3.966, -2.635, [0, 0, 0, 1, 0, 0, 0]]]
    """
    
    # Define residue classes with updated indices
    residue_classes = {'ALA': 1,'GLY': 1,'ILE': 1,'LEU': 1,'MET': 1,'VAL': 1,
                       'PHE': 2,'TRP': 2,'TYR': 2,
                       'ASN': 3,'CYS': 3,'GLN': 3,'PRO': 3,'SER': 3,'THR': 3,
                       'ASP': 4,'GLU': 4,
                       'ARG': 5,'HIS': 5,'LYS': 5,
                       'UNK': 6,'NON': 6}

    translated_data = []
    for _olfr in res_data:
        _olfr_res = []
        for _res in _olfr: 
            x, y, z = map(float, _res[3:6])  # Extract coordinates and convert to float
            residue = _res[1]  # Extract residue name
            cls = residue_classes.get(residue, 6)  # Default to class 6 if residue not found
            class_vector = [0] * 7  # Initialize all-zero vector
            class_vector[cls] = 1  # Set appropriate class index to 1
            _olfr_res.append([x, y, z, class_vector])  # Append the result
        translated_data.append(_olfr_res)
    return translated_data


# Categorize amino acids by key binding-relevant properties
AA_PROPERTIES = {
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
AA_PROPERTY_LOOKUP = create_reverse_property_mapping(AA_PROPERTIES)

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

def encode_residues_for_voxel(res_data):
    """
    Translates residue coordinates into a list of coordinates and their 
    one-hot encoded property vectors.
    
    Parameters:
    -----------
    res_data : list
        A list of lists, where each sublist contains residue information in the format:
        [residue_number, residue_name, atom_name, x, y, z]
    
    Returns:
    --------
    list
        A list of lists, where each sublist contains:
        [x, y, z, one_hot_encoded_vector]
    """
    translated_data = []
    for _olfr in res_data:
        _olfr_res = []
        for _res in _olfr:
            x, y, z = map(float, _res[3:6])  # Extract coordinates and convert to float
            residue = _res[1]  # Extract residue name
            
            # Create property vector
            property_vector = [0]  # Cavity flag
            property_vector.extend(one_hot_encode_aa_properties(residue))
            
            _olfr_res.append([x, y, z, property_vector])  # Append the result
        translated_data.append(_olfr_res)
    
    return translated_data

def voxelize_cavity(
    cavity_coords=None, 
    residue_coords=None, 
    resolution=1, 
    encode_method='aa_properties'
):
    """
    Voxelizes 3D coordinates by creating a voxel grid representation.
    
    Parameters:
    -----------
    cavity_coords : dict or list, optional
        3D coordinates for cavities
    residue_coords : dict or list, optional
        Residue interaction data
    resolution : float, default 1
        Size of each voxel
    encode_method : str, default 'aa_properties'
        Method of encoding residue properties
    
    Returns:
    --------
    tuple: 
        - List of voxel grids
        - Tuple of grid shape (X, Y, Z)
    """

    # Convert dict to list if necessary
    cavity_coords = list(cavity_coords.values()) if isinstance(cavity_coords, dict) else (cavity_coords or [])
    residue_coords = list(residue_coords.values()) if isinstance(residue_coords, dict) else (residue_coords or [])

    # Prepare residue coordinates based on encoding method
    if encode_method == 'aa_properties': 
        residue_coords_class = encode_residues_for_voxel(residue_coords) if residue_coords else None
    else: 
        residue_coords_class = _res_coord_voxel_prep(residue_coords) if residue_coords else None

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

    # Compute grid boundaries
    min_coords = np.min(all_coords, axis=0)
    max_coords = np.max(all_coords, axis=0)

    # Define voxel grid shape
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

    for i in range(num_iterations):
        # Initialize voxel grid with dynamic vector length
        voxel_grid = np.zeros((*grid_shape, vector_length), dtype=int)

        # Collect coordinates for current iteration
        coords_data = []
        
        # Add residue coordinates if available
        if residue_coords_class and i < len(residue_coords_class):
            coords_data.extend(residue_coords_class[i])
        
        # Add cavity coordinates if available
        if cavity_coords and i < len(cavity_coords):
            cavity_flag_vector = [1] + [0] * (vector_length - 1)
            coords_data.extend([
                [_coord[0], _coord[1], _coord[2], cavity_flag_vector] 
                for _coord in cavity_coords[i]
            ])

        # Map coordinates to voxel grid
        for point in coords_data:
            grid_x = int((point[0] - min_coords[0]) // resolution)
            grid_y = int((point[1] - min_coords[1]) // resolution)
            grid_z = int((point[2] - min_coords[2]) // resolution)
            voxel_grid[grid_x, grid_y, grid_z] = point[3]

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