import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm

import color_function as cf

import pyKVFinder
from typing import List, Dict

def load_pdb_coordinates(pdb_file):
    """
    Extracts atomic coordinates and converts amino acid sequence to single-letter notation.
    
    :param pdb_file: Path to the PDB file.
    :return: A tuple of (coordinates, sequence in single-letter notation).
    
    :usage
    coords, backbone, seq = load_pdb_coordinates(pdb_file)

    """
    aa_three_to_one = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }
    coords = []
    backbone = []
    sequence = []
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                parts = line.split()
                # Extract coordinates
                x = float(parts[6])
                y = float(parts[7])
                z = float(parts[8])
                coords.append([x,y,z])
                if " CA " in line:  # Select alpha-carbon atoms
                    backbone.append([x, y, z])
                    # Extract residue name
                    residue = parts[3]  # Residue name
                    single_letter = aa_three_to_one.get(residue, "X")  # Use "X" for unknown residues
                    sequence.append(single_letter)
    return np.array(coords), np.array(backbone), "".join(sequence)

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

    for cavnum, key in enumerate(cavities_coords, start=2):
        indexes = _grid2indexes(results.cavities, cavnum)
        coords = _indexes2coord(indexes, results._step, results._vertices)
        cavities_coords[key] = coords
        
    # Get cavities center and limits
    center = [{key: value.mean(axis=0)} for key, value in cavities_coords.items()]
    minmax = {
        key: [  # KAA, KAB, ...
            value.min(axis=0),  # [xmin, ymin, zmin]
            value.max(axis=0),  # [xmax, ymax, zmax]
        ]
        for key, value in cavities_coords.items()
    }

    return cavities_coords, center, minmax


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

