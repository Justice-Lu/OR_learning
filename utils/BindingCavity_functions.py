import numpy as np
import os 
import plotly.graph_objects as go
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, Delaunay

import color_function as cf
from pdb_functions import load_pdb_coordinates

import pyKVFinder
from typing import List, Dict


AA_THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

# def load_pdb_coordinates(pdb_file, single_aa_name=True):
#     """
#     Extracts atomic coordinates and converts amino acid sequence to single-letter notation.
    
#     :param pdb_file: Path to the PDB file.
#     :return: A tuple of (coordinates, sequence in single-letter notation).
    
#     :usage
#     coords, backbone, seq = load_pdb_coordinates(pdb_file)

#     """

#     coords = []
#     backbone = []
#     sequence = []
#     with open(pdb_file, 'r') as file:
#         for line in file:
#             if line.startswith("ATOM"):
#                 parts = line.split()
#                 # Extract coordinates
#                 x = float(parts[6])
#                 y = float(parts[7])
#                 z = float(parts[8])
#                 coords.append([x,y,z])
#                 if " CA " in line:  # Select alpha-carbon atoms
#                     backbone.append([x, y, z])
#                     # Extract residue name
#                     residue = parts[3]  # Residue name
#                     if single_aa_name: 
#                         single_letter = AA_THREE_TO_ONE.get(residue, "X")  # Use "X" for unknown residues
#                         sequence.append(single_letter)
#                     else: 
#                         sequence.append((residue, parts[5])) # Store aa and residue number in tuple 
#     if single_aa_name: 
#         return np.array(coords), np.array(backbone), "".join(sequence)
#     else: 
#         return np.array(coords), np.array(backbone), sequence

def single_Olfr_cavity(arr, color_dict=None, trace_size=1, trace_opacity=0.3):
    """
    Plots a 3D array as a scatter plot with categorical colors.

    Parameters:
        arr (np.ndarray): A 3D numpy array to be plotted.
        color_dict (dict): A dictionary mapping unique values to specific colors. If None, default colors are used.
        trace_size (float): The size of the scatter points.
        trace_opacity (float): The opacity of the scatter points.

    Returns:
        None: Displays the 3D scatter plot.
    """
    # Get the coordinates (x, y, z) and values
    x, y, z = np.indices(arr.shape)
    values = arr.flatten()

    # Filter out the -1 values (empty space)
    mask = values != -1
    x_filtered = x.flatten()[mask]
    y_filtered = y.flatten()[mask]
    z_filtered = z.flatten()[mask]
    values_filtered = values[mask]

    # Get the unique values (excluding -1)
    unique_values = np.unique(values_filtered)

    # Create a color map based on the unique values
    num_colors = len(unique_values)

    # Use provided color_dict or generate a default color map
    if color_dict is None:
        color_map = cf.distinct_colors(unique_values)
    else:
        assert len(color_dict) >= num_colors, "Color dictionary must have enough colors for all unique values."
        color_map = color_dict

    # Create the 3D scatter plot
    fig = go.Figure()

    # Add a scatter trace for each unique value to create a legend
    for val in unique_values:
        # Get the points for this value
        mask_val = values_filtered == val
        x_val = x_filtered[mask_val]
        y_val = y_filtered[mask_val]
        z_val = z_filtered[mask_val]

        # Add the trace to the figure
        fig.add_trace(go.Scatter3d(
            x=x_val,
            y=y_val,
            z=z_val,
            mode='markers',
            marker=dict(
                size=trace_size,
                color=color_map[val],  # Color for this value
                opacity=trace_opacity
            ),
            name=str(val)  # Add the value to the legend
        ))

    # Add layout details
    fig.update_layout(
        width=600, height=600,
        scene=dict(
            xaxis=dict(range=[0, arr.shape[0]], title='X', visible=False, showbackground=False),
            yaxis=dict(range=[0, arr.shape[1]], title='Y', visible=False, showbackground=False),
            zaxis=dict(range=[0, arr.shape[2]], title='Z', visible=False, showbackground=False)
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        legend_title="Values"
    )

    # Show the plot
    return fig


def multi_Olfr_cavity(pyKVFinder_result_list, color_dict=None, trace_size=1, trace_opacity=0.3):
    """
    Plots a 3D array as a scatter plot with categorical colors.

    Parameters:
        arr (np.ndarray): A 3D numpy array to be plotted.
        color_dict (dict): A dictionary mapping unique values to specific colors. If None, default colors are used.
        trace_size (float): The size of the scatter points.
        trace_opacity (float): The opacity of the scatter points.

    Returns:
        None: Displays the 3D scatter plot.
    """
    
    # Create a color map based on the num Olfr
    num_colors = len(pyKVFinder_result_list)
    
    # Use provided color_dict or generate a default color map
    if color_dict is None:
        color_map = cf.distinct_colors(list(range(num_colors)))
    else:
        assert len(color_dict) >= num_colors, "Color dictionary must have enough colors for all unique values."
        color_map = color_dict
    
    
    # Create the 3D scatter plot
    fig = go.Figure()
    
    for i, _result in enumerate(pyKVFinder_result_list): 
        arr = _result.cavities # Extracts cavities from pyKVFindfer results
        # Get the coordinates (x, y, z) and values
        x, y, z = np.indices(arr.shape)
        values = arr.flatten()

        # Filter for the 0 values (protein)
        mask = values == 0
        prot_x = x.flatten()[mask]
        prot_y = y.flatten()[mask]
        prot_z = z.flatten()[mask]
        prot_opacity = np.max([trace_opacity - 0.3, 0.1]) # prot_opacity is lesser than binding cavity
        
        fig.add_trace(go.Scatter3d(
            x=prot_x,
            y=prot_y,
            z=prot_z,
            mode='markers',
            marker=dict(
                size=trace_size,
                color=color_map[i],  # Color for this value
                opacity=prot_opacity
            ),
            name=f"{str(i)} prot"  # Add the value to the legend
        ))
        
        # Filter for the >1 values (cavities)
        mask = values > 1
        cav_x = x.flatten()[mask]
        cav_y = y.flatten()[mask]
        cav_z = z.flatten()[mask]
        
        fig.add_trace(go.Scatter3d(
            x=cav_x,
            y=cav_y,
            z=cav_z,
            mode='markers',
            marker=dict(
                size=trace_size,
                color=color_map[i],  # Color for this value
                opacity=trace_opacity
            ),
            name=f"{str(i)} cavity"  # Add the value to the legend
        ))


    # Add layout details
    fig.update_layout(
        width=600, height=600,
        scene=dict(
            xaxis=dict(range=[0, arr.shape[0]], title='X', visible=False, showbackground=False),
            yaxis=dict(range=[0, arr.shape[1]], title='Y', visible=False, showbackground=False),
            zaxis=dict(range=[0, arr.shape[2]], title='Z', visible=False, showbackground=False)
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        legend_title="Values"
    )

    # Show the plot
    return fig



# Adapted from (https://github.com/LBC-LNBio/pyKVFinder/issues/110)
# Series of functions to get the cavity coordinates instead of embedded in grid form
def _grid2indexes(cavities: np.ndarray, cavnum: int) -> np.ndarray:
    indexes = np.argwhere(cavities == cavnum)
    return indexes


def _indexes2coord(
    indexes: np.ndarray, step: float, vertices: np.ndarray
) -> np.ndarray:
    from pyKVFinder.grid import _get_sincos

    # P1, P2, P3, P4 (origin, x-axis, y-axis, z-axis)
    P1, P2, P3, P4 = vertices

    # Calculate sin and cos for each axis
    sincos = _get_sincos(vertices)

    # Convert grid to 3D Cartesian coordinates
    xaux, yaux, zaux = (indexes * step).T

    x = (
        (xaux * sincos[3])
        + (yaux * sincos[0] * sincos[2])
        - (zaux * sincos[1] * sincos[2])
        + P1[0]
    )
    y = (yaux * sincos[1]) + (zaux * sincos[0]) + P1[1]
    z = (
        (xaux * sincos[2])
        - (yaux * sincos[0] * sincos[3])
        + (zaux * sincos[1] * sincos[3])
        + P1[2]
    )

    # Prepare 3D coordinates
    coords = np.array([x, y, z]).T

    return coords


def grid2coords(results: pyKVFinder.pyKVFinderResults) -> dict:
    # Prepare dictionary to store cavities coordinates
    cavities_coords = {key: [] for key in results.residues.keys()}
    cavities_surface_coords = {key: [] for key in results.residues.keys()}
    
    for cavnum, key in enumerate(cavities_coords, start=2):
        # Save cavity coordinates
        indexes = _grid2indexes(results.cavities, cavnum)
        coords = _indexes2coord(indexes, results._step, results._vertices)
        cavities_coords[key] = coords
        # Save surface coordinates (Useful later when defining canonical binding cavity space)
        indexes = _grid2indexes(results.surface, cavnum)
        coords = _indexes2coord(indexes, results._step, results._vertices)
        cavities_surface_coords[key] = coords
        
    # Get cavities center and limits
    center = [{key: value.mean(axis=0)} for key, value in cavities_coords.items()]
    minmax = {
        key: [  # KAA, KAB, ...
            value.min(axis=0),  # [xmin, ymin, zmin]
            value.max(axis=0),  # [xmax, ymax, zmax]
        ]
        for key, value in cavities_coords.items()
    }

    return cavities_coords, cavities_surface_coords, center, minmax


def coords2pdb(coords: Dict[str, np.ndarray], filename: str = "cavity.pdb") -> None:
    with open(filename, "w") as f:
        i = 0
        for key, coords in coords.items():
            for coord in coords:
                i += 1
                f.write(
                    "ATOM  {:5d}  H   {:3s}   259    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00            \n".format(
                        i, key, coord[0], coord[1], coord[2]
                    )
                )
                

def _get_atomic_information(
    residues: Dict[str, List[str]], cavtag: str, atomic: np.ndarray
) -> np.ndarray:
    # Get atomic information from residues
    resatomic = np.array(["_".join(item[0:3]) for item in residues[cavtag]])

    # Extract atominfo from atomic
    atominfo = np.asarray(
        ([[f"{atom[0]}_{atom[1]}_{atom[2]}", atom[3]] for atom in atomic[:, :4]])
    )

    # Get coordinates of residues
    indexes = np.in1d(atominfo[:, 0], resatomic)

    return atomic[indexes]


def res2atomic(results: pyKVFinder.pyKVFinderResults, atomic: np.ndarray) -> Dict[str, np.ndarray]:
    # Prepare dictionary to store residues coordinates
    residues_coords = {key: [] for key in results.residues.keys()}

    for cavtag in residues_coords.keys():
        # Get coordinates of residues
        residues_coords[cavtag] = _get_atomic_information(
            results.residues, cavtag, atomic
        )

    return residues_coords

def run_pyKVFinder_workflow(pdb_files, dict_keys=None, parameter_set=None, cavity_identity=False):
    """
    Runs the pyKVFinder standard workflow for a list of PDB files and extracts cavity, cavity surface,
    and interacting residue coordinates.

    :param pdb_files: List of PDB file paths.
    :param dict_keys: Custom dictionary keys for PDB file identifiers (optional).
    :param parameter_set: Dictionary of parameters for pyKVFinder (optional, only modifies given parameters).
    :return: Tuple (cav_coords, cavsurf_coords, res_coords), where each is a dictionary 
             mapping PDB identifiers to coordinate lists.
    """
    
    # Default parameter set
    default_params = {"probe_in": 1.6, "probe_out": 4.0, "removal_distance": 3.0, "volume_cutoff": 20.0}
    # default_params = {"probe_in": 1.0, "probe_out": 3.0, "removal_distance": 2.0, "volume_cutoff": 20.0}

    # If user provides some parameters, override only those while keeping the rest as default
    params = {**default_params, **(parameter_set or {})}

    cav_coords = {}
    cavsurf_coords = {}
    res_coords = {}

    for i, _pdb in enumerate(pdb_files): 
        # Define dictionary key
        if dict_keys is None:
            _olfr = _pdb.split('/')[-1].replace('.pdb', '').replace('_tmaligned', '')
            _olfr = f"{_olfr.split('.')[0]}_{str(i+1)}" if '.' in _olfr else _olfr
        else:
            _olfr = dict_keys[i]

        # Run pyKVFinder with selected parameters
        results = pyKVFinder.run_workflow(
            _pdb,
            probe_in=params["probe_in"],
            probe_out=params["probe_out"],
            removal_distance=params["removal_distance"],
            volume_cutoff=params["volume_cutoff"]
        )
        
        atomic_data = pyKVFinder.read_pdb(_pdb)
        results_coord = grid2coords(results)

        if cavity_identity: # Extract in pyKV cavity form. contains cavity separation
            cav_coords[_olfr] = results_coord[0] # Extract cavity coordinates
            cavsurf_coords[_olfr] = results_coord[1] # Extract cavity surface coordinates
            
            res_coords_dict = res2atomic(results, atomic_data) # Extract cavity interacting residue coordinates
            res_coords[_olfr] = {cav_res: (res_coords_dict[cav_res][:, [0, 2, 3, 4, 5, 6]]) for cav_res in res_coords_dict }

        else: # Extract in list form. 
            cav_coords[_olfr] = [coord for cavity in results_coord[0].values() for coord in cavity]
            cavsurf_coords[_olfr] = [coord for surface in results_coord[1].values() for coord in surface]
            
            res_coords_dict = res2atomic(results, atomic_data)
            res_coords[_olfr] = [
                list(x) for x in set(
                    tuple(entry) for res in res_coords_dict for entry in res_coords_dict[res][:, [0, 2, 3, 4, 5, 6]].tolist()
                )
            ]

    return cav_coords, cavsurf_coords, res_coords

# Functions below for defining canonical binding cavity and identifying residues 
def define_binding_cavity_zone(bc_cavsurf_coords, expansion_distance=3.0):
    """
    Defines the overall binding cavity zone by identifying the largest cavity for each OR, 
    expanding it, and superimposing the expanded zones.

    :param bc_cavsurf_coords: 
        Dictionary where keys are OR names and values are numpy arrays of shape (N, 3) representing cavity surface coordinates.
    :param expansion_distance: 
        Float, the distance to expand the largest cavity surface points.
    
    :return: 
        A dictionary with the following keys:
        - "largest_cavity_coords": A dictionary of largest cavity coordinates for each OR.
        - "expanded_coords": A dictionary of expanded cavity zones for each OR.
    """

    largest_cavity_coords = {}
    expanded_coords = {}

    # Find the largest cavity for each OR and expand it
    for _Or in bc_cavsurf_coords.keys():
        # Cluster points to identify distinct cavities
        dbscan = DBSCAN(eps=2, min_samples=5)
        labels = dbscan.fit_predict(bc_cavsurf_coords[_Or])

        # Calculate cavity sizes
        cavity_sizes = {label: np.sum(labels == label) for label in set(labels) if label != -1}

        # Find the largest cavity
        largest_cavity_label = max(cavity_sizes, key=cavity_sizes.get)
        largest_cavity_coords[_Or] = np.array(bc_cavsurf_coords[_Or])[labels == largest_cavity_label]
        
        # Expand each surface point outward by the specified distance
        expanded_zone = []
        for point in largest_cavity_coords[_Or]:
            for dx in np.arange(-expansion_distance, expansion_distance + 1, 1):
                for dy in np.arange(-expansion_distance, expansion_distance + 1, 1):
                    for dz in np.arange(-expansion_distance, expansion_distance + 1, 1):
                        # Only add points within the desired expansion radius
                        if np.sqrt(dx**2 + dy**2 + dz**2) <= expansion_distance:
                            expanded_zone.append(point + np.array([dx, dy, dz]))

        # Store unique expanded coordinates for the current OR
        expanded_coords[_Or] = np.unique(expanded_zone, axis=0)

    return expanded_coords, largest_cavity_coords


def filter_coordinates_within_cavity(cavity_zone, cavity_coordinates, residue_coordinates=None, 
                                     filter_cutoff = 0.9):
    """
    Filters 3D coordinates or residue coordinates to include only those that lie within the convex hull 
    of a specified cavity zone. If residue coordinates are used, only their 3D coordinates are retained.

    Automatically determines if input is residue or coordinate format based on array shape.

    :param cavity_zone: 
        A numpy array of shape (N, 3) representing the 3D points defining the cavity zone.
    :param cavity_coordinates: 
        - If a list/array:
            - A numpy array of shape (M, 3) representing 3D points.
            - A numpy array of shape (M, 6) where the last three columns represent 3D coordinates.
        - If a dictionary:
            - Keys represent cluster names (e.g., cavity labels), and values are numpy arrays.
    :param residue_coordinates:
        - Optional, same format as `cavity_coordinates` but representing residue positions.
        - If dictionary format is used, it must match `cavity_coordinates` keys.
    
    :return: 
        - If input was a list/array, returns filtered numpy array.
        - If input was a dictionary, returns (filtered_cavity_clusters, filtered_residue_coords),
          where `filtered_residue_coords` contains only the last three columns of residue coordinates.
    """

    # Validate input dimensions
    if cavity_zone.shape[1] != 3:
        raise ValueError("cavity_zone must have shape (N, 3).")

    # Create a Delaunay triangulation for point-in-hull checks
    hull_delaunay = Delaunay(cavity_zone)

    # Determine input format
    cavity_is_dict = isinstance(cavity_coordinates, dict)
    residue_is_dict = isinstance(residue_coordinates, dict) if residue_coordinates is not None else False

    if cavity_is_dict and residue_is_dict:
        # Dictionary format processing
        filtered_cavity_clusters = []
        filtered_cavity_keys = []
        
        for cavity_key, cavity_coords in cavity_coordinates.items():
            inside_mask = hull_delaunay.find_simplex(cavity_coords) >= 0
            
            if np.mean(inside_mask) >= filter_cutoff:  # Keep only clusters where all points are inside
                filtered_cavity_clusters.extend(cavity_coords)
                filtered_cavity_keys.append(cavity_key)

        # Keep only residues that match retained cavity cluster keys, but return only coordinates
        filtered_residue_coords = [
            residue_coordinates[key]
            for key in filtered_cavity_keys if key in residue_coordinates
        ]

        return np.array(filtered_cavity_clusters), np.concatenate(filtered_residue_coords) if filtered_residue_coords else np.empty((0, 3))

    elif cavity_is_dict or residue_is_dict:
        raise ValueError("Both cavity_coordinates and residue_coordinates must be dictionaries if one of them is.")

    else:
        # Array format processing: determine if it's a residue format (6 columns) or cavity format (3 columns)
        if cavity_coordinates.shape[1] == 6:
            coords = cavity_coordinates
        elif cavity_coordinates.shape[1] == 3:
            coords = cavity_coordinates.astype(float)
        else:
            raise ValueError("Invalid format for cavity_coordinates. Must have 3 or 6 columns.")

        # Apply filtering
        inside_mask = hull_delaunay.find_simplex(coords) >= 0
        return coords[inside_mask]