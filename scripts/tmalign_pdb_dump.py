import os
import numpy as np
from tmtools import tm_align


def load_pdb_coordinates(pdb_file):
    """
    Extracts atomic coordinates and converts amino acid sequence to single-letter notation.
    
    :param pdb_file: Path to the PDB file.
    :return: A tuple of (coordinates, sequence in single-letter notation).
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


def align_and_save(ref_pdb, target_pdb, output_pdb, align_with='backbone'):
    """
    Align a target PDB to a reference PDB and save the aligned structure.
    
    :param ref_pdb: Path to the reference PDB file.
    :param target_pdb: Path to the target PDB file.
    :param output_pdb: Path to save the aligned PDB file.
    :param align_with: Either 'backbone' or 'full_coords' for alignment.
    """
    assert align_with in ['backbone', 'full_coords'], "Please choose either 'backbone' or 'full_coords' for alignment"
    
    # Load coordinates and sequences
    coords1, backbone1, seq1 = load_pdb_coordinates(ref_pdb)
    coords2, backbone2, seq2 = load_pdb_coordinates(target_pdb)
    
    # Perform alignment
    if align_with == 'backbone':
        aligned_result = tm_align(backbone1, backbone2, seq1, seq2)
    elif align_with == 'full_coords':
        aligned_result = tm_align(coords1, coords2, seq1, seq2)
    
    # Rotate and translate all atom coordinates
    transformed_coords = np.dot(coords2, aligned_result.u)

    # Save the rotated PDB file
    with open(target_pdb, 'r') as file_in, open(output_pdb, 'w') as file_out:
        atom_index = 0  # Index to track the transformed coordinates
        for line in file_in:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Use split to extract parts of the line
                parts = line.split()
                
                # Handle discrepancies in indexing:
                # Assume fixed PDB field order: [Record, Serial, Name, AltLoc, ResName, Chain, ResNum, X, Y, Z, Occupancy, TempFactor, Element, Charge]
                serial = int(parts[1])
                name = parts[2]
                alt_loc = parts[3] if len(parts[3]) == 1 else ""
                res_name = parts[3 if alt_loc == "" else 4]
                chain = parts[4 if alt_loc == "" else 5]
                res_num = int(parts[5 if alt_loc == "" else 6])
                x = float(parts[6 if alt_loc == "" else 7])
                y = float(parts[7 if alt_loc == "" else 8])
                z = float(parts[8 if alt_loc == "" else 9])
                occupancy = float(parts[9 if alt_loc == "" else 10]) if len(parts) > (9 if alt_loc == "" else 10) else 1.0
                temp_factor = float(parts[10 if alt_loc == "" else 11]) if len(parts) > (10 if alt_loc == "" else 11) else 0.0
                element = parts[11 if alt_loc == "" else 12] if len(parts) > (11 if alt_loc == "" else 12) else " "
                charge = parts[12 if alt_loc == "" else 13] if len(parts) > (12 if alt_loc == "" else 13) else " "
                
                # Get the transformed coordinates
                x, y, z = transformed_coords[atom_index]
                atom_index += 1
                
                # Write the updated line in PDB format
                file_out.write(
                    f"{line[:6]}{serial:5d} {name:<4}{alt_loc}{res_name:>3} {chain}"
                    f"{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{occupancy:6.2f}{temp_factor:6.2f}          {element:>2}{charge:>2}\n"
                )
            else:
                # Write non-ATOM lines unchanged
                file_out.write(line)
    
    print(f"Aligned PDB saved to: {output_pdb}")

def batch_align(ref_pdb, input_dir, output_dir):
    """
    Align all PDB files in a directory to the reference PDB file.
    
    :param ref_pdb: Path to the reference PDB file.
    :param input_dir: Directory containing target PDB files.
    :param output_dir: Directory to save the aligned PDB files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for pdb_file in os.listdir(input_dir):
        if pdb_file.endswith(".pdb"):
            target_pdb = os.path.join(input_dir, pdb_file)
            output_pdb = os.path.join(output_dir, f"{os.path.splitext(pdb_file)[0]}_tmaligned.pdb")
            align_and_save(ref_pdb, target_pdb, output_pdb)

def main():
    # Configuration
    ref_pdb = "/data/jlu/AF_files/AF_pdb/Olfr1377.pdb"             # Path to the reference PDB file
    input_dir = "/data/jlu/AF_files/AF_pdb"      # Directory containing PDB files to align
    output_dir = "/data/jlu/AF_files/AF_tmaligned_pdb"   # Directory to save the aligned PDB files

    # Perform batch alignment
    batch_align(ref_pdb, input_dir, output_dir)
    print(f"Batch alignment completed. Aligned files are saved in: {output_dir}")

if __name__ == "__main__":
    main()