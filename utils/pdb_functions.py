import numpy as np 
import pandas as pd
import os 
import re 
import tempfile


from tmtools import tm_align 


AA_THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
            

# def load_pdb_coordinates(pdb_file, single_aa_name=True):
#     """
#     Extracts atomic coordinates from a PDB file, ensuring strict adherence to the PDB format first.
#     Falls back to a space-split method if the format is incorrect.

#     Parameters:
#         pdb_file (str): Path to the PDB file.
#         single_aa_name (bool): Whether to return the sequence in single-letter format.

#     Returns:
#         tuple: (coordinates, backbone, sequence)
#             - coordinates (numpy.ndarray): All atomic coordinates.
#             - backbone (numpy.ndarray): Alpha-carbon (CA) backbone coordinates.
#             - sequence (str or list): Amino acid sequence in single-letter notation or tuples (res_name, res_num).
#     """
#     coords = []
#     backbone = []
#     sequence = []

#     with open(pdb_file, 'r') as file:
#         for line in file:
#             if line.startswith("ATOM"):
#                 try:
#                     # Strict PDB format parsing
#                     serial = int(line[6:11])   # Atom serial number
#                     name = line[12:16].strip() # Atom name
#                     alt_loc = line[16:17]      # Alternate location indicator
#                     res_name = line[17:20].strip() # Residue name
#                     chain = line[21:22]        # Chain identifier
#                     res_num = int(line[22:26]) # Residue sequence number
#                     x = float(line[30:38])     # X coordinate
#                     y = float(line[38:46])     # Y coordinate
#                     z = float(line[46:54])     # Z coordinate

#                 except ValueError:
#                     # If strict parsing fails, fall back to split-based parsing
#                     parts = line.split()
#                     if len(parts) < 9:
#                         continue  # Skip malformed lines

#                     try:
#                         name = parts[2]  # Atom name
#                         x = float(parts[6])
#                         y = float(parts[7])
#                         z = float(parts[8])
#                         res_name = parts[3]
#                         res_num = parts[5] if len(parts) > 5 else "?"
#                     except (IndexError, ValueError):
#                         continue  # Skip malformed lines

#                 coords.append([x, y, z])

#                 # Alpha-carbon backbone selection
#                 if name == "CA":  # Use 'name' instead of 'parts'
#                     backbone.append([x, y, z])
                    
#                     # Extract sequence
#                     if single_aa_name:
#                         single_letter = AA_THREE_TO_ONE.get(res_name, "X")  # "X" for unknown residues
#                         sequence.append(single_letter)
#                     else:
#                         sequence.append((res_name, res_num))  # Store (residue name, residue number)

#     if single_aa_name:
#         return np.array(coords), np.array(backbone), "".join(sequence)
#     else:
#         return np.array(coords), np.array(backbone), sequence
    
def read_pdb(pdb_file, chain_id=None, include_ligands=True):
    """
    Reads a PDB file and extracts atom information while ensuring strict PDB format adherence.
    Optionally filters atoms by a specific chain ID and includes/excludes ligands (HETATM lines).

    :param pdb_file: Path to the PDB file.
    :param chain_id: Specific chain ID to extract (default: None, keeps all chains).
    :param include_ligands: Boolean flag to include ligands (HETATM) or not (default: True).
    :return: List of atom dictionaries containing parsed PDB information.
    """
    atoms = []
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or (include_ligands and line.startswith("HETATM")):
                try:
                    chain = line[21:22]
                    if chain_id is None or chain in chain_id:
                        atom = {
                            "serial": int(line[6:11]),
                            "name": line[12:16].strip(),
                            "alt_loc": line[16:17],
                            "res_name": line[17:20].strip(),
                            "chain": chain,
                            "res_num": int(line[22:26]),
                            "x": float(line[30:38]),
                            "y": float(line[38:46]),
                            "z": float(line[46:54]),
                            "occupancy": float(line[54:60]),
                            "temp_factor": float(line[60:66]),
                            "element": line[76:78].strip(),
                            "charge": line[78:80].strip()
                        }
                        atoms.append(atom)
                except ValueError:
                    print(f"Warning: Could not parse line: {line.strip()}")
    return atoms


def write_pdb(atoms, output_pdb):
    """
    Writes atom information to a PDB file following strict formatting guidelines.
    
    :param atoms: List of atom dictionaries containing parsed PDB information.
    :param output_pdb: Path to save the formatted PDB file.
    """
    with open(output_pdb, 'w') as file_out:
        for atom in atoms:
            head = 'ATOM' if atom['res_name'] in AA_THREE_TO_ONE.keys() else 'HETATM'
            
            file_out.write(
                f"{head:<6}{atom['serial']:5d}  {atom['name']:<3}{atom['alt_loc']}{atom['res_name']:>3} {atom['chain']}"
                f"{atom['res_num']:4d}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                f"{atom['occupancy']:6.2f}{atom['temp_factor']:6.2f}          {atom['element']:>2}{atom['charge']:>2}\n"
            )


def load_pdb_coordinates(pdb_file, single_aa_name=True, chain_id=None, keep_ligand=False):
    """
    Extracts atomic coordinates, separates ligand coordinates, and converts amino acid sequence to single-letter notation.

    :param pdb_file: Path to the PDB file.
    :param single_aa_name: Whether to return the sequence in single-letter notation.
    :param chain_id: Specific chain ID to extract (default: None, keeps all chains).
    :param keep_ligand: Whether to separate ligand coordinates from the main protein structure.
    :return: A tuple of (coordinates, backbone coordinates, sequence, ligand coordinates if keep_ligands=True).
    """
    atoms = read_pdb(pdb_file, chain_id=chain_id, include_ligands=keep_ligand)
    coords = []
    backbone = []
    sequence = []
    ligands = []

    for atom in atoms:
        x, y, z = atom['x'], atom['y'], atom['z']
        
        if atom['res_name'] in AA_THREE_TO_ONE.keys():  # Check if it's a standard amino acid
            coords.append([x, y, z])
            
            if atom['name'] == 'CA':  # Select alpha-carbon atoms
                backbone.append([x, y, z])
                res_name = atom['res_name']
                
                if single_aa_name:
                    single_letter = AA_THREE_TO_ONE.get(res_name, "X")  # Use "X" for unknown residues
                    sequence.append(single_letter)
                else:
                    sequence.append((res_name, atom['res_num']))  # Store aa and residue number
        else:
            ligands.append([x, y, z])  # Store ligand coordinates separately
    
    if single_aa_name:
        sequence = "".join(sequence)

    if keep_ligand:
        return np.array(coords), np.array(backbone), sequence, np.array(ligands)
    else:
        return np.array(coords), np.array(backbone), sequence

def tmalign_pdb(ref_pdb, target_pdb, 
                output_pdb = None, 
                align_with='backbone',
                save_pdb = True, 
                return_coords = False):
    """
    Aligns a target PDB to a reference PDB and saves the aligned structure.
    
    :param ref_pdb: Path to the reference PDB file.
    :param target_pdb: Path to the target PDB file.
    :param output_pdb: Path to save the aligned PDB file.
    :param align_with: Either 'backbone' or 'full_coords' for alignment.
    """
    assert align_with in ['backbone', 'full_coords'], "Please choose either 'backbone' or 'full_coords' for alignment"
    if not return_coords and save_pdb: 
        assert output_pdb is not None, "Please provide output_pdb if saving pdb"
    
    coords1, backbone1, seq1 = load_pdb_coordinates(ref_pdb)
    coords2, backbone2, seq2, ligand = load_pdb_coordinates(target_pdb, keep_ligand=True)
    
    ref_atoms = backbone1 if align_with == 'backbone' else coords1
    tgt_atoms = backbone2 if align_with == 'backbone' else coords2
    
    centroid_ref = np.mean(ref_atoms, axis=0)
    centroid_tgt = np.mean(tgt_atoms, axis=0)
    
    ref_atoms_centered = ref_atoms - centroid_ref
    tgt_atoms_centered = tgt_atoms - centroid_tgt
    
    aligned_result = tm_align(ref_atoms_centered, tgt_atoms_centered, seq1, seq2)
    transformed_coords = np.dot(coords2 - centroid_tgt, aligned_result.u) + centroid_ref
    
    if len(ligand) > 0: 
        transformed_ligand = np.dot(ligand - centroid_tgt, aligned_result.u) + centroid_ref
        atoms = read_pdb(target_pdb, include_ligands=True)
    else:  
        atoms = read_pdb(target_pdb, include_ligands=False)
    # return transformed_coords, atoms # FOR DEBUGGING
    for i, atom in enumerate(atoms):
        if i < len(transformed_coords): 
            atom['x'], atom['y'], atom['z'] = transformed_coords[i]
        else: 
            atom['x'], atom['y'], atom['z'] = transformed_ligand[i-len(transformed_coords)]
        
    if save_pdb:
        write_pdb(atoms, output_pdb)
        print(f"Aligned PDB saved to: {output_pdb}")
        
    if return_coords: 
    # return rotated_coords, rotated_backbone
        return np.dot(coords2 - centroid_tgt, aligned_result.u) + centroid_ref, \
               np.dot(backbone2 - centroid_tgt, aligned_result.u) + centroid_ref
    
# A quick fix, as a lot of notebooks still uses align_and_save call
align_and_save = tmalign_pdb

    
def fix_pdb_format(input_pdb, output_pdb=None, chain_id=None, keep_ligand=True):
    """
    Fixes the formatting of a PDB file to adhere strictly to PDB format specifications.
    First attempts direct line parsing; if it fails, falls back to part-based parsing.
    Allows filtering by chain ID and optional separation of ligands.

    :param input_pdb: Path to the input PDB file (old format with inconsistent spacing).
    :param output_pdb: Path to save the fixed PDB file. If None, overwrites the input file.
    :param chain_id: Specific chain ID to keep (default: None, keeps all chains).
    :param keep_ligands: Whether to include ligand entries in the output PDB.
    """
    if output_pdb is None:
        output_pdb = input_pdb  # Overwrite the original file if no output path is provided

    fixed_lines = []
    
    with open(input_pdb, 'r') as file_in:
        for line in file_in:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    # Attempt direct parsing using fixed positions
                    serial = int(line[6:11].strip())
                    name = line[12:16].strip()
                    alt_loc = line[16:17]
                    res_name = line[17:20].strip()
                    chain = line[21:22]
                    res_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    occupancy = float(line[54:60].strip()) if line[54:60].strip() else 1.00
                    temp_factor = float(line[60:66].strip()) if line[60:66].strip() else 0.00
                    element = line[76:78].strip() if len(line) > 76 else "  "
                    charge = line[78:80].strip() if len(line) > 78 else "  "

                except ValueError:
                    # If direct parsing fails, attempt dynamic parsing
                    try:
                        parts = line.split()
                        serial = int(parts[1])  
                        name = parts[2]  
                        if len(parts[3]) == 1:  # Detect if there's an altLoc indicator
                            alt_loc = parts[3]
                            res_name = parts[4]
                            chain = parts[5]
                            res_num = int(parts[6])
                            x = float(parts[7])
                            y = float(parts[8])
                            z = float(parts[9])
                            occupancy = float(parts[10]) if len(parts) > 10 else 1.00
                            temp_factor = float(parts[11]) if len(parts) > 11 else 0.00
                            element = parts[12] if len(parts) > 12 else "  "
                            charge = parts[13] if len(parts) > 13 else "  "
                        else:  
                            alt_loc = " "  # No alternate location indicator
                            res_name = parts[3]
                            chain = parts[4]
                            res_num = int(parts[5])
                            x = float(parts[6])
                            y = float(parts[7])
                            z = float(parts[8])
                            occupancy = float(parts[9]) if len(parts) > 9 else 1.00
                            temp_factor = float(parts[10]) if len(parts) > 10 else 0.00
                            element = parts[11] if len(parts) > 11 else "  "
                            charge = parts[12] if len(parts) > 12 else "  "

                    except (ValueError, IndexError):
                        # If both methods fail, skip the line
                        print(f"Warning: Could not parse line, skipping: {line.strip()}")
                        fixed_lines.append(line)  # Preserve for debugging
                        continue  

                # Apply chain and ligand filtering
                if chain_id is not None and chain not in chain_id:
                    continue
                if not keep_ligand and res_name not in AA_THREE_TO_ONE.keys():
                    continue

                # Format the line properly following strict PDB specifications
                fixed_line = (
                    f"{line[:6]}{serial:5d}  {name:<3}{alt_loc}{res_name:>3} {chain}"
                    f"{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{occupancy:6.2f}{temp_factor:6.2f}          {element:>2}{charge:>2}\n"
                )
                fixed_lines.append(fixed_line)

            else:
                fixed_lines.append(line)  # Preserve non-ATOM/HETATM lines

    # Save the fixed PDB file
    with open(output_pdb, 'w') as file_out:
        file_out.writelines(fixed_lines)
    
    print(f"Fixed PDB saved to: {output_pdb}")


def filter_cif_by_chain(input_file, output_file, chain_id='A', use_auth_chain=True):
    """
    Filter a .cif file to keep only atoms from a specific chain while preserving metadata.
    
    Args:
        input_file (str): Path to input .cif file
        output_file (str): Path to output .cif file
        chain_id (str): Chain identifier to keep (default: 'A')
        use_auth_chain (bool): If True, filter by auth_asym_id, otherwise by label_asym_id
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if starting the atom_site loop
            if line == "loop_":
                # Check if the next line starts an atom site section
                if i+1 < len(lines) and (lines[i+1].strip() == "_atom_site.group_PDB" or 
                                          lines[i+1].strip().startswith("_atom_site.")):
                    # We're in the atom section
                    outfile.write(lines[i])  # Write the "loop_" line
                    i += 1
                    
                    # Read and store all header lines
                    header_lines = []
                    column_indices = {}
                    header_idx = 0
                    
                    while i < len(lines) and lines[i].strip().startswith("_atom_site."):
                        header_line = lines[i].strip()
                        header_lines.append(lines[i])
                        column_indices[header_idx] = header_line
                        header_idx += 1
                        i += 1
                    
                    # Find the index of the chain ID columns
                    chain_col_idx = None
                    auth_chain_col_idx = None
                    
                    for idx, col_name in column_indices.items():
                        if "_atom_site.label_asym_id" in col_name:
                            chain_col_idx = idx
                        elif "_atom_site.auth_asym_id" in col_name:
                            auth_chain_col_idx = idx
                    
                    # Write all header lines to output
                    for header in header_lines:
                        outfile.write(header)
                    
                    # Now process atom lines
                    while i < len(lines) and not (lines[i].strip() == "" or 
                                                 lines[i].strip().startswith("#") or 
                                                 lines[i].strip().startswith("loop_")):
                        # Split the line into columns
                        columns = lines[i].split()
                        
                        if len(columns) >= max(chain_col_idx or 0, auth_chain_col_idx or 0) + 1:
                            # Determine which chain ID to use for filtering
                            if use_auth_chain and auth_chain_col_idx is not None:
                                current_chain = columns[auth_chain_col_idx]
                            elif chain_col_idx is not None:
                                current_chain = columns[chain_col_idx]
                            else:
                                # If can't determine chain, write the line anyway
                                outfile.write(lines[i])
                                i += 1
                                continue
                            
                            # Write the line if it matches the requested chain
                            if current_chain in chain_id:
                                outfile.write(lines[i])
                        else:
                            # Line doesn't have enough columns, write it anyway (might be a comment)
                            outfile.write(lines[i])
                        i += 1
                else:
                    # Not the atom section, just write the line
                    outfile.write(lines[i])
                    i += 1
            else:
                # Not in atom section, write the line
                outfile.write(lines[i])
                i += 1

def cif_to_pdb(cif_file, pdb_out_path, remove_H = True):
    """
    Converts a CIF file to a properly formatted PDB file.

    :param cif_file: Path to the input CIF file.
    :param pdb_out_path: Path to save the converted PDB file.
    """
    atoms = []

    with open(cif_file, 'r') as file:
        for line in file:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()                
                atom = {
                        "serial": int(parts[1]),
                        "name": parts[3],
                        "alt_loc": " ",
                        "res_name": parts[5][0:3],
                        "chain": parts[6],
                        "res_num": int(parts[8]) if str(parts[8]).isdigit() else 999, # Hard code ligands as residue 999 
                        "x": float(parts[10]),
                        "y": float(parts[11]),
                        "z": float(parts[12]),
                        "occupancy": float(parts[13]) if len(parts) > 13 else 1.00,
                        "temp_factor": float(parts[14]) if len(parts) > 14 else 0.00,
                        "element":  parts[2][0] if len(parts) > 2 else " ", # Guess element from atom name
                        "charge": " "  # No charge information in CIF
                    }
                
                # Skip hydrogen atoms if remove_H is True
                if remove_H and atom["element"] == "H":
                    continue
                
                # If the atom is a ligand, then assign chain as 'A'
                atom["chain"] = 'A' if atom["res_num"] == 999 else atom["chain"]
                
                # Use the existing write_pdb function
                atoms.append(atom)

    # Save the converted PDB file
    write_pdb(atoms, pdb_out_path)

    print(f"Converted PDB saved to: {pdb_out_path}")
    

# Homology Model via modeller 

from modeller import *
from modeller.automodel import *

def run_modeller_homology(aln_file, 
                          template_pdb,
                          target_name,
                          atom_files_directory=['.'],
                          output_dir=None, 
                          num_models=3):
    """
    Runs homology modeling using Modeller and organizes the output files.

    :param aln_file: Path to the alignment file (.fa) containing sequence information.
    :param template_pdb: Name of the template structure (without file extension).
    :param target_name: Name of the target protein to be modeled.
    :param output_dir: Directory to store output files (default: target_name + "_modeller").
    :param num_models: Number of models to generate (default: 3).
    :return: Path to the output directory containing homology models.
    """

    class MyModel(AutoModel):
        pass  # No special assumptions for now

    env = Environ()
    env.io.atom_files_directory = atom_files_directory

    # Initialize homology modeling
    a = MyModel(env, alnfile=aln_file, knowns=template_pdb, sequence=target_name)
    a.starting_model = 1
    a.ending_model = num_models
    a.make()

    # Define output directory
    output_dir = output_dir or f"{target_name}_modeller"
    os.makedirs(output_dir, exist_ok=True)

    # Organize output files
    for _file in os.listdir():
        if os.path.isfile(_file) and _file.startswith(target_name):
            if _file.endswith('.pdb'):  # Move PDB files
                os.rename(_file, os.path.join(output_dir, f'HM_{_file}'))
            elif _file != aln_file:  # Remove unnecessary files
                os.remove(_file)

    print(f"Output files saved in: {output_dir}")

def write_modeller_fa(tgt_aligned_seq, ref_aligned_seq, 
                      ref_pdb, 
                      labels=None, 
                      output_file="alignment.fa"):
    """
    Writes a Modeller-compatible alignment file (.fa) for homology modeling.

    :param tgt_aligned_seq: Target sequence (string) in aligned format.
    :param ref_aligned_seq: Reference sequence (string) in aligned format.
    :param labels: List containing two labels [target_label, reference_label].
    :param output_file: Name of the output file (default: 'alignment.fa').
    :raises ValueError: If sequences are not the same length.
    """
    if len(tgt_aligned_seq) != len(ref_aligned_seq):
        raise ValueError("Aligned sequences must have the same length.")

    def get_residue_indices(sequence):
        """Get the first and last residue number excluding gaps ('-')."""
        residues = [i + 1 for i, res in enumerate(sequence) if res != '-']
        if residues:
            return residues[0], residues[-1]  # Start, End
        return " ", " "  # Handle case where sequence is all gaps

    # Extract labels
    target_label, ref_label = labels or ['target', 'template']

    # Compute start and end residue positions
    tgt_start, tgt_end = get_residue_indices(tgt_aligned_seq)
    
    # Structure pdb requires the position to line up with pdb. Read from pdb instead
    ref_atoms = read_pdb(ref_pdb)
    ref_start, ref_end = ref_atoms[0]['res_num'], ref_atoms[-1]['res_num']

    fa_content = f""">P1;{target_label}
sequence:{target_label}:{tgt_start}: :{tgt_end}: ::: 0.00: 0.00
{tgt_aligned_seq}*

>P1;{ref_label}.pdb
structureX:{ref_label}.pdb:{ref_start}:A:{ref_end}:A::: 0.00: 0.00
{ref_aligned_seq}*
"""

    with open(output_file, "w") as f:
        f.write(fa_content)

    print(f"Alignment file written to {output_file}")


