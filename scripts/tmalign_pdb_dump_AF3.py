import os
import sys 
import numpy as np
from tmtools import tm_align

sys.path.insert(0, '/mnt/data2/Justice/OR_learning/utils')
import pdb_functions as pu

"""
The script reads in and aligns .pdb 
"""

def batch_align(ref_pdb, output_dir, separate_OR = True ):
    """
    Align all PDB files in a directory to the reference PDB file.
    
    :param ref_pdb: Path to the reference PDB file.
    :param input_dir: Directory containing target PDB files.
    :param output_dir: Directory to save the aligned PDB files.
    """
    # os.makedirs(output_dir, exist_ok=True)
    if separate_OR: 
        for OR_dir in os.listdir(output_dir):
            for root, dirs, files in os.walk(os.path.join(output_dir, OR_dir)):
                for _file in files: 
                    if _file.endswith('.pdb'):
                        target_pdb = os.path.join(root, _file)
                        output_pdb = os.path.join(root, f"{os.path.splitext(_file)[0]}_tmaligned.pdb")
                        
                        if not os.path.exists(output_pdb):
                            pu.tmalign_pdb(ref_pdb, target_pdb, output_pdb)
                        else: 
                            print(f"Skipping . . . {output_pdb} already exist")
    else: 
        for root, dirs, files in os.walk(output_dir):
            for _file in files: 
                if _file.endswith('.pdb'):
                    target_pdb = os.path.join(root, _file)
                    output_pdb = os.path.join(root, f"{os.path.splitext(_file)[0]}_tmaligned.pdb")
                if not os.path.exists(output_pdb):
                    pu.tmalign_pdb(ref_pdb, target_pdb, output_pdb)
                else: 
                    print(f"Skipping . . . {output_pdb} already exist")
            
def batch_cif_to_pdb(input_dir, out_dir, separate_OR = True): 
    
    OVERWRITE = False

    for OR_dir in os.listdir(input_dir):
        # OR_name = OR_dir.split('_')[0]
        OR_name = OR_dir.split('_')[1]
        
        for root, dirs, files in os.walk(os.path.join(input_dir, OR_dir)):
            for i, _seed_dir in enumerate(dirs):
                files = os.listdir(os.path.join(root, _seed_dir))
                for file in files: 
                    if file.endswith(".cif"):
                        cif_path = os.path.join(root, _seed_dir, file)
                        # Build pdb paths in same folder as cif
                        if separate_OR: 
                            pdb_path = os.path.join(out_dir, OR_dir, f'{OR_name}_{i}.pdb')
                            # nolig_pdb_path = os.path.join(out_dir, OR_dir, f'{OR_name}_{i}_nolig.pdb')
                            
                            if not os.path.isdir(os.path.join(out_dir, OR_dir)): os.mkdir(os.path.join(out_dir, OR_dir))
                        else: 
                            pdb_path = os.path.join(out_dir, f'{OR_name}_{i}.pdb') # save without OR_dir
                            if not os.path.isdir(os.path.join(out_dir)): os.mkdir(os.path.join(out_dir))
                        
                         

                        if OVERWRITE or not os.path.exists(pdb_path):
                            print(f"Converting {cif_path} → {pdb_path}")
                            pu.cif_to_pdb(cif_path, pdb_path)

                            # Fix format, chain A only
                            pu.fix_pdb_format(pdb_path, pdb_path, chain_id='A')

                            # # Also make no-ligand version
                            # pu.fix_pdb_format(pdb_path, nolig_pdb_path, keep_ligand=False)
                        else:
                            print(f"{pdb_path} already exists, skipping.")  

def main():
    # Configuration
    ref_pdb = "/mnt/data2/Justice/AF_files/AF_tmaligned_pdb/Olfr1377_tmaligned.pdb" # Path to the reference PDB file
    
    # input_dir = "/mnt/data2/Justice/AF3_files/AF3_out/AF3_OR_Ga/"  # Directory containing PDB files to align
    # output_dir = "/mnt/data2/Justice/AF3_files/AF3_out/AF3_OR_tmaligned/" # Directory to save the aligned PDB files
    
    # input_dir = "/mnt/data2/Justice/AF3_files/AF3_out/mouseOR_golf/"  # Directory containing PDB files to align
    # output_dir = "/mnt/data2/Justice/AF3_files/AF3_out/mouseOR_golf_pdb/" # Directory to save the aligned PDB files

    input_dir = "/mnt/data2/Justice/AF3_files/AF3_out/humanOR_golf/"  # Directory containing PDB files to align
    output_dir = "/mnt/data2/Justice/AF3_files/AF3_out/humanOR_golf_pdb/" # Directory to save the aligned PDB files
    
    SEPARATE_OR_DIR = False 
    
    batch_cif_to_pdb(input_dir, output_dir, separate_OR=SEPARATE_OR_DIR)
    
    
    # Perform batch alignment
    batch_align(ref_pdb, output_dir, separate_OR = SEPARATE_OR_DIR)
    print(f"Batch alignment completed. Aligned files are saved in: {output_dir}")

if __name__ == "__main__":
    main()